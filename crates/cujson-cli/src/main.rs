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
#[command(name = "cujson", version, about = "GPU JSON parsing, from the command line")]
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

fn cmd_info() -> Result<(), CliError> {
    #[cfg(not(feature = "cuda"))]
    {
        println!("built without CUDA");
        Err(cujson::Error::CudaNotCompiled.into())
    }
    #[cfg(feature = "cuda")]
    {
        let info = cujson::cuda_info()?;
        println!("cuJSON-rs {}", env!("CARGO_PKG_VERSION"));
        println!("compiled archs: {}", info.compiled_archs);
        println!("CUDA runtime version: {}", info.runtime_version);
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
        let warm = if times.len() > 1 { &times[1..] } else { &times[..] };
        let mut sorted: Vec<_> = warm.to_vec();
        sorted.sort();
        let min = sorted.first().copied().unwrap_or_default();
        let median = sorted[sorted.len() / 2];
        println!("parse time: min={min:?} median={median:?} (n={})", sorted.len());
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
    writeln!(out, "idx\toffset\tchar\tpair")?;
    let t = &doc.tape;
    for i in 0..t.len() {
        let offset = t.structural[i];
        let ch = if offset >= 1 && (offset as usize - 1) < bytes.len() {
            bytes[offset as usize - 1] as char
        } else {
            '?'
        };
        writeln!(out, "{i}\t{offset}\t{ch}\t{}", t.pair_pos[i])?;
    }
    Ok(())
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
        let v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| CliError::Other(e.to_string()))?;
        std::hint::black_box(&v);
        serde_times.push(start.elapsed());
    }

    fn summarize(mut times: Vec<std::time::Duration>) -> (std::time::Duration, std::time::Duration) {
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
