//! `cujson` CLI (docs/plan/07-cli.md). Built on the public `cujson` API
//! only — no tape-comparison logic re-implemented here, see
//! `cujson::tape::diff_tapes`.

use std::io::Write;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, Subcommand};

mod verify;

/// Exit code used for "CUDA not available" (no device, no driver, or not
/// compiled in) as distinct from a verification FAIL (which uses 1).
const EXIT_NO_CUDA: u8 = 2;

#[derive(Parser)]
#[command(
    name = "cujson",
    version,
    about = "GPU JSON parsing, from the command line"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// CUDA runtime/device info: version, compiled archs, visible devices.
    Info,
    /// Parse a file on the GPU and print a summary or pointer values.
    Parse {
        file: PathBuf,
        #[arg(long)]
        lines: bool,
        #[arg(long, default_value_t = 256)]
        chunk_mb: usize,
        #[arg(long = "pointer")]
        pointers: Vec<String>,
        #[arg(long)]
        time: bool,
        #[arg(long, default_value_t = 1)]
        repeat: usize,
    },
    /// Dump the tape as TSV: idx  offset  char  pair.
    Tape {
        file: PathBuf,
        #[arg(long)]
        lines: bool,
        /// Use the CPU reference builder instead of the GPU.
        #[arg(long)]
        cpu: bool,
    },
    /// Compare GPU output against the CPU reference and serde_json.
    Verify {
        /// If omitted, runs the bundled fixtures.
        file: Option<PathBuf>,
        #[arg(long)]
        lines: bool,
        #[arg(long)]
        chunk_bytes: Option<usize>,
    },
    /// Wall-clock cuJSON vs serde_json (vs simd-json if built with that feature).
    Bench {
        file: PathBuf,
        #[arg(long, default_value_t = 5)]
        repeat: usize,
    },
}

fn main() -> ExitCode {
    // Rust ignores SIGPIPE, so writing to a closed pipe (`cujson tape F | head`)
    // makes `println!` panic. Restore the default action so the process ends
    // quietly, as other Unix CLIs do.
    #[cfg(unix)]
    // SAFETY: runs first in main, before any other thread exists.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    run()
}

fn run() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Info => cmd_info(),
        Command::Parse {
            file,
            lines,
            chunk_mb,
            pointers,
            time,
            repeat,
        } => cmd_parse(&file, lines, chunk_mb, &pointers, time, repeat),
        Command::Tape { file, lines, cpu } => cmd_tape(&file, lines, cpu),
        Command::Verify {
            file,
            lines,
            chunk_bytes,
        } => return verify::run(file, lines, chunk_bytes),
        Command::Bench { file, repeat } => cmd_bench(&file, repeat),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(CliError::NoCuda(msg)) => {
            eprintln!("{msg}");
            ExitCode::from(EXIT_NO_CUDA)
        }
        Err(CliError::Other(msg)) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

enum CliError {
    NoCuda(String),
    Other(String),
}

impl From<cujson::Error> for CliError {
    fn from(e: cujson::Error) -> Self {
        let message = e.to_string();
        match e {
            cujson::Error::CudaNotCompiled => CliError::NoCuda(
                "cujson-cli was built without the `cuda` feature (rebuild with --features cuda)"
                    .to_string(),
            ),
            cujson::Error::NoDevice => {
                CliError::NoCuda("no CUDA device visible on this host".to_string())
            }
            // A CUDA runtime/driver call itself failed (e.g. no driver
            // installed, driver too old) — treated the same as "no
            // device" for exit-code purposes (docs/plan/07-cli.md).
            cujson::Error::Cuda { .. } => CliError::NoCuda(format!("CUDA unavailable: {message}")),
            _ => CliError::Other(message),
        }
    }
}

impl From<std::io::Error> for CliError {
    fn from(e: std::io::Error) -> Self {
        CliError::Other(e.to_string())
    }
}

/// Plain-words hint for a `cujson::Error` from `cuda_info()`/`parse*`,
/// beyond the error's own `Display` — the specific cases a user hits on a
/// real GPU host: no driver at all, a driver too old for this binary's
/// CUDA runtime, or a driver with no device visible. `None` when the
/// error doesn't warrant an extra hint (unrecognized code, or CUDA not
/// compiled in — already self-explanatory).
#[cfg(feature = "cuda")]
pub(crate) fn cuda_error_hint(err: &cujson::Error) -> Option<String> {
    let code = match err {
        cujson::Error::Cuda { code, .. } => *code,
        cujson::Error::NoDevice => {
            return Some(
                "driver present but no CUDA device visible (check CUDA_VISIBLE_DEVICES / container GPU passthrough)"
                    .to_string(),
            );
        }
        _ => return None,
    };
    match code {
        35 => {
            let driver = cujson::driver_version();
            if driver == 0 {
                Some("no NVIDIA driver found".to_string())
            } else {
                let runtime = cujson::runtime_version().unwrap_or(0);
                Some(format!(
                    "driver supports CUDA {}.{} but this binary was built with CUDA runtime {}.{}; update the driver or rebuild with an older toolkit",
                    driver / 1000,
                    (driver % 1000) / 10,
                    runtime / 1000,
                    (runtime % 1000) / 10,
                ))
            }
        }
        100 => Some(
            "driver present but no CUDA device visible (check CUDA_VISIBLE_DEVICES / container GPU passthrough)"
                .to_string(),
        ),
        _ => None,
    }
}

fn cmd_info() -> Result<(), CliError> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("built without CUDA");
        Err(cujson::Error::CudaNotCompiled.into())
    }
    #[cfg(feature = "cuda")]
    {
        let info = match cujson::cuda_info() {
            Ok(i) => i,
            Err(e) => {
                if let Some(hint) = cuda_error_hint(&e) {
                    eprintln!("{hint}");
                }
                return Err(e.into());
            }
        };
        println!("cuJSON-rs {}", env!("CARGO_PKG_VERSION"));
        println!("compiled archs: {}", info.compiled_archs);
        println!("CUDA runtime version: {}", info.runtime_version);
        println!("CUDA driver version: {}", info.driver_version);
        if info.devices.is_empty() {
            println!("no CUDA devices visible");
        } else {
            for d in &info.devices {
                println!("device {}: {}", d.index, d.name);
            }
        }
        Ok(())
    }
}

fn cmd_parse(
    file: &PathBuf,
    lines: bool,
    chunk_mb: usize,
    pointers: &[String],
    time: bool,
    repeat: usize,
) -> Result<(), CliError> {
    let bytes = std::fs::read(file)?;
    let repeat = repeat.max(1);
    let mut times = Vec::with_capacity(repeat);
    let mut last_doc = None;
    for _ in 0..repeat {
        let start = Instant::now();
        let doc = if lines {
            cujson::parse_lines(
                &bytes,
                cujson::LinesOptions {
                    chunk_bytes: chunk_mb * 1024 * 1024,
                },
            )?
        } else {
            cujson::parse(&bytes)?
        };
        times.push(start.elapsed());
        last_doc = Some(doc);
    }
    let doc = last_doc.expect("repeat >= 1");

    if pointers.is_empty() {
        println!(
            "{} bytes, tape length {}, depth {}",
            bytes.len(),
            doc.tape.len(),
            doc.depth()
        );
    } else {
        for p in pointers {
            match doc.pointer(p) {
                Some(node) => println!("{p}: {}", node.to_value()),
                None => println!("{p}: <not found>"),
            }
        }
    }

    if time {
        // Exclude the first run (warm-up / context init) when repeat > 1.
        let warm = if times.len() > 1 {
            &times[1..]
        } else {
            &times[..]
        };
        let mut sorted: Vec<_> = warm.to_vec();
        sorted.sort();
        let min = sorted.first().copied().unwrap_or_default();
        let median = sorted[sorted.len() / 2];
        println!(
            "parse time: min={min:?} median={median:?} (n={})",
            sorted.len()
        );
    }
    Ok(())
}

fn cmd_tape(file: &PathBuf, lines: bool, cpu: bool) -> Result<(), CliError> {
    let bytes = std::fs::read(file)?;
    let doc = if cpu {
        let mode = if lines {
            cujson::cpu::Mode::Lines
        } else {
            cujson::cpu::Mode::Standard
        };
        cujson::cpu::parse(&bytes, mode).map_err(|e| CliError::Other(e.to_string()))?
    } else if lines {
        cujson::parse_lines(&bytes, cujson::LinesOptions::default())?
    } else {
        cujson::parse(&bytes)?
    };

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let result = (|| -> std::io::Result<()> {
        writeln!(out, "idx\toffset\tchar\tpair")?;
        let t = &doc.tape;
        for i in 0..t.len() {
            let offset = t.structural[i];
            let ch = tape_char(i, t.len(), offset, &bytes);
            writeln!(out, "{i}\t{offset}\t{ch}\t{}", t.pair_pos[i])?;
        }
        Ok(())
    })();
    ignore_broken_pipe(result)?;
    Ok(())
}

/// Mirrors `Document::get_char` (`crates/cujson/src/tape/document.rs`) —
/// the artificial wrapper entries at index `0`/`len-1` read as `[`/`]`,
/// and an unescaped `\n` structural entry reads as `,` — so `cujson tape`'s
/// dump matches `tape/FORMAT.md` §2 exactly, since it's the tool used to
/// debug a `verify` failure against that same spec.
fn tape_char(idx: usize, total: usize, offset: i32, input: &[u8]) -> char {
    if idx == 0 {
        return '[';
    }
    if idx + 1 == total {
        return ']';
    }
    let pos = offset - 1;
    if pos < 0 || pos as usize >= input.len() {
        return '?';
    }
    let c = input[pos as usize];
    if c == b'\n' { ',' } else { c as char }
}

/// Stdout being closed early (e.g. `cujson tape FILE | head`) is a normal
/// exit for a streaming dump command, not a program error.
fn ignore_broken_pipe(result: std::io::Result<()>) -> Result<(), CliError> {
    match result {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => Err(e.into()),
    }
}

fn cmd_bench(file: &PathBuf, repeat: usize) -> Result<(), CliError> {
    let bytes = std::fs::read(file)?;
    let repeat = repeat.max(1);

    // cuJSON: includes H2D copy, note that clearly.
    let mut cujson_times = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let start = Instant::now();
        let doc = cujson::parse(&bytes)?;
        std::hint::black_box(&doc);
        cujson_times.push(start.elapsed());
    }

    let mut serde_times = Vec::with_capacity(repeat);
    for _ in 0..repeat {
        let start = Instant::now();
        let v: serde_json::Value =
            serde_json::from_slice(&bytes).map_err(|e| CliError::Other(e.to_string()))?;
        std::hint::black_box(&v);
        serde_times.push(start.elapsed());
    }

    fn summarize(
        mut times: Vec<std::time::Duration>,
    ) -> (std::time::Duration, std::time::Duration) {
        times.sort();
        let min = times[0];
        let median = times[times.len() / 2];
        (min, median)
    }

    let (cmin, cmed) = summarize(cujson_times);
    let (smin, smed) = summarize(serde_times);
    println!("cuJSON (includes H2D copy): min={cmin:?} median={cmed:?}");
    println!("serde_json:                 min={smin:?} median={smed:?}");

    #[cfg(feature = "simd-json")]
    {
        let mut simd_times = Vec::with_capacity(repeat);
        for _ in 0..repeat {
            let mut buf = bytes.clone();
            let start = Instant::now();
            let v: simd_json::OwnedValue =
                simd_json::to_owned_value(&mut buf).map_err(|e| CliError::Other(e.to_string()))?;
            std::hint::black_box(&v);
            simd_times.push(start.elapsed());
        }
        let (mmin, mmed) = summarize(simd_times);
        println!("simd-json:                   min={mmin:?} median={mmed:?}");
    }
    #[cfg(not(feature = "simd-json"))]
    {
        println!("simd-json: not built (enable with --features simd-json)");
    }

    Ok(())
}
