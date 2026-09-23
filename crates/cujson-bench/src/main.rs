mod data;
mod engines;
mod walk;

use std::time::Duration;

use clap::{Parser, Subcommand};
use engines::{Engine, Level, Tape, run_batch, run_pipeline};

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(clap::Args, Clone)]
struct Input {
    /// Parquet file with a JSON-string column
    #[arg(
        long,
        default_value = "crates/cujson-bench/data/chunk_0-00283-of-00546.parquet"
    )]
    input: String,
    #[arg(long, default_value = "claims")]
    column: String,
    /// Rows are grouped into JSON Lines batches of at most this many MiB
    #[arg(long, default_value_t = 256)]
    batch_mb: usize,
    /// Rayon threads (default: all cores)
    #[arg(long)]
    threads: Option<usize>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Time engines over the whole column
    Run {
        #[command(flatten)]
        input: Input,
        #[arg(long, value_enum, value_delimiter = ',', default_values_t = [Engine::Simd, Engine::SimdPar, Engine::SimdBuf, Engine::SimdBufPar, Engine::CujsonNode, Engine::CujsonNodePar, Engine::CujsonVisit, Engine::CujsonVisitPar, Engine::CujsonPipe])]
        engines: Vec<Engine>,
        /// Tape source for the cujson-* engines
        #[arg(long, value_enum, default_value_t = Tape::Gpu)]
        tape: Tape,
        #[arg(long, value_enum, value_delimiter = ',', default_values_t = [Level::Parse, Level::Walk])]
        levels: Vec<Level>,
        #[arg(long, default_value_t = 5)]
        reps: usize,
        #[arg(long, default_value_t = 2)]
        warmup: usize,
        /// One JSON object per result instead of a table
        #[arg(long)]
        json: bool,
        /// Give the cujson-* engines their input in pinned memory (copied once, untimed)
        #[arg(long)]
        pinned_input: bool,
        /// Threads parsing on the GPU at once in cujson-pipe
        #[arg(long, default_value_t = 1)]
        gpu_threads: usize,
    },
    /// Parse the same batch repeatedly, printing RSS and free GPU memory per iteration
    Leak {
        #[command(flatten)]
        input: Input,
        #[arg(long, value_enum, default_value_t = Engine::CujsonNode)]
        engine: Engine,
        #[arg(long, default_value_t = 30)]
        iters: usize,
    },
    /// Check that every engine produces the same per-row checksum
    Verify {
        #[command(flatten)]
        input: Input,
        #[arg(long, default_value_t = 2000)]
        rows: usize,
    },
}

fn proc_kb(key: &str) -> Option<u64> {
    std::fs::read_to_string("/proc/self/status")
        .ok()?
        .lines()
        .find_map(|l| {
            l.strip_prefix(key)?
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse()
                .ok()
        })
}

fn reset_peak_rss() {
    let _ = std::fs::write("/proc/self/clear_refs", "5");
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn load(input: &Input, max_rows: Option<usize>) -> data::Corpus {
    if let Some(t) = input.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok();
    }
    data::load(&input.input, &input.column, input.batch_mb << 20, max_rows).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(2)
    })
}

fn main() {
    match Cli::parse().cmd {
        Cmd::Run {
            input,
            engines,
            levels,
            tape,
            reps,
            warmup,
            json,
            pinned_input,
            gpu_threads,
        } => bench(
            &input,
            &engines,
            &levels,
            tape,
            reps,
            warmup,
            json,
            pinned_input,
            gpu_threads,
        ),
        Cmd::Verify { input, rows } => verify(&input, rows),
        Cmd::Leak {
            input,
            engine,
            iters,
        } => leak(&input, engine, iters),
    }
}

#[allow(clippy::too_many_arguments)]
fn bench(
    input: &Input,
    engines: &[Engine],
    levels: &[Level],
    tape: Tape,
    reps: usize,
    warmup: usize,
    json: bool,
    pinned_input: bool,
    gpu_threads: usize,
) {
    let mut corpus = load(input, None);
    if pinned_input {
        for b in &mut corpus.batches {
            b.pinned = Some(
                cujson::PinnedBuffer::from_slice(&b.bytes).unwrap_or_else(|e| {
                    eprintln!("pinned input: {e}");
                    std::process::exit(2)
                }),
            );
        }
    }
    let (rows, bytes) = (corpus.rows(), corpus.bytes());
    let gb = bytes as f64 / 1e9;
    eprintln!(
        "{}: {} file rows -> {} rows ({} null, {} empty, {} with raw newline skipped), {:.1} MB in {} batches, mean {:.0} B/row, max {} B, {} threads",
        input.input,
        corpus.file_rows,
        rows,
        corpus.null_rows,
        corpus.empty_rows,
        corpus.newline_rows,
        bytes as f64 / 1e6,
        corpus.batches.len(),
        bytes as f64 / rows as f64,
        corpus.max_row_bytes,
        rayon::current_num_threads(),
    );
    let baseline_kb = proc_kb("VmRSS:").unwrap_or(0);
    let mut reference: Vec<(Level, u64, u64)> = vec![];
    let mut medians: Vec<(Engine, Level, f64)> = vec![];
    if levels.contains(&Level::Walk) {
        // Every walk result is checked against simd-json's, even when it is the only engine run.
        let (mut r, mut h) = (0, 0u64);
        for b in &corpus.batches {
            let run =
                run_batch(Engine::SimdPar, Level::Walk, tape, b).expect("simd-json reference");
            r += run.rows;
            h = h.wrapping_add(run.hash);
        }
        reference.push((Level::Walk, r, h));
    }

    for &engine in engines {
        if engine.is_cujson()
            && tape == Tape::Gpu
            && let Err(e) = cujson::parse_lines(b"{}\n", cujson::LinesOptions::default())
        {
            eprintln!("{engine:?}: unavailable ({e})");
            continue;
        }
        for &level in levels {
            if engine == Engine::CujsonPipe && level == Level::Parse {
                continue;
            }
            reset_peak_rss();
            let mut runs: Vec<Vec<f64>> = vec![];
            let mut names: Vec<&'static str> = vec![];
            let mut totals = vec![];
            let mut check = (0u64, 0u64);
            for rep in 0..warmup + reps {
                let mut phases: Vec<f64> = vec![];
                let (mut r, mut h) = (0, 0u64);
                let mut total = Duration::ZERO;
                let results: Vec<Result<engines::Run, String>> = if engine == Engine::CujsonPipe {
                    vec![run_pipeline(&corpus.batches, tape, gpu_threads)]
                } else {
                    corpus
                        .batches
                        .iter()
                        .map(|b| run_batch(engine, level, tape, b))
                        .collect()
                };
                for result in results {
                    match result {
                        Ok(run) => {
                            if phases.is_empty() {
                                phases = vec![0.0; run.phases.len()];
                                names = run.phases.iter().map(|p| p.0).collect();
                            }
                            for (i, p) in run.phases.iter().enumerate() {
                                phases[i] += p.1.as_secs_f64();
                            }
                            total += run.total();
                            r += run.rows;
                            h = h.wrapping_add(run.hash);
                        }
                        Err(e) => {
                            eprintln!("{engine:?}/{level:?}: {e}");
                            std::process::exit(1);
                        }
                    }
                }
                check = (r, h);
                if rep >= warmup {
                    runs.push(phases);
                    totals.push(total.as_secs_f64());
                }
            }
            if level == Level::Walk {
                match reference.iter().find(|x| x.0 == level) {
                    None => reference.push((level, check.0, check.1)),
                    Some(x) if (x.1, x.2) != check => {
                        eprintln!(
                            "{engine:?}: CHECKSUM MISMATCH vs first engine ({:?} vs {:?}) - run `verify`",
                            (x.1, x.2),
                            check
                        );
                        std::process::exit(1);
                    }
                    _ => {}
                }
            }
            let mut t = totals.clone();
            let med = median(&mut t);
            medians.push((engine, level, med));
            let phase_med: Vec<f64> = (0..names.len())
                .map(|i| median(&mut runs.iter().map(|r| r[i]).collect::<Vec<_>>()))
                .collect();
            let peak_over_kb = proc_kb("VmHWM:").unwrap_or(0).saturating_sub(baseline_kb);
            if json {
                let ph: serde_json::Map<_, _> = names
                    .iter()
                    .zip(&phase_med)
                    .map(|(n, s)| (n.to_string(), (*s).into()))
                    .collect();
                println!(
                    "{}",
                    serde_json::json!({
                        "engine": format!("{engine:?}"), "level": format!("{level:?}"), "rows": rows, "bytes": bytes,
                        "median_s": med, "min_s": t[0], "max_s": t[t.len() - 1], "gb_per_s": gb / med,
                        "rows_per_s": rows as f64 / med, "phases_s": ph, "peak_rss_over_corpus_mb": peak_over_kb / 1024,
                        "checksum": format!("{:016x}", check.1), "reps": reps,
                    })
                );
            } else {
                let ph: Vec<String> = names
                    .iter()
                    .zip(&phase_med)
                    .map(|(n, s)| format!("{n} {s:.3}s"))
                    .collect();
                println!(
                    "{:<11} {:<5} {:>7.3}s [{:.3}-{:.3}] {:>6.2} GB/s {:>9.0} rows/s  +{} MB RSS  | {}",
                    format!("{engine:?}"),
                    format!("{level:?}"),
                    med,
                    t[0],
                    t[t.len() - 1],
                    gb / med,
                    rows as f64 / med,
                    peak_over_kb / 1024,
                    ph.join(", ")
                );
            }
        }
    }
    let find = |e: Engine, l: Level| medians.iter().find(|m| m.0 == e && m.1 == l).map(|m| m.2);
    if let (Some(simd), Some(pipe)) = (
        find(Engine::SimdPar, Level::Walk),
        find(Engine::CujsonPipe, Level::Walk),
    ) {
        let ratio = simd / pipe;
        if json {
            println!(
                "{}",
                serde_json::json!({
                    "summary": "simd-par vs cujson-pipe, whole file, walk level",
                    "simd_par_s": simd, "cujson_pipe_s": pipe, "speedup": ratio,
                    "tape": format!("{tape:?}"),
                })
            );
        } else {
            let verdict = if ratio >= 1.0 {
                format!("cujson-pipe is {ratio:.2}x faster")
            } else {
                format!("cujson-pipe is {:.2}x slower", 1.0 / ratio)
            };
            let note = if tape == Tape::Cpu {
                " (--tape cpu: the cuJSON tape is built on the CPU, not a GPU result)"
            } else {
                ""
            };
            println!(
                "overall (walk, whole file): simd-par {simd:.3}s, cujson-pipe {pipe:.3}s: {verdict}{note}"
            );
        }
    }
}

fn row_hashes_simd(bytes: &[u8]) -> Vec<u64> {
    let mut work = bytes.to_vec();
    work.split_mut(|&b| b == b'\n')
        .filter(|r| !r.is_empty())
        .map(|r| walk::walk_simd(&simd_json::to_borrowed_value(r).expect("simd-json parse")))
        .collect()
}

fn verify(input: &Input, max_rows: usize) {
    let corpus = load(input, Some(max_rows));
    let mut sets: Vec<(&str, Vec<u64>)> = vec![];
    let mut simd = vec![];
    let mut cpu = vec![];
    let mut gpu = vec![];
    let mut gpu_err = None;
    for b in &corpus.batches {
        simd.extend(row_hashes_simd(&b.bytes));
        let d = cujson::cpu::parse(&b.bytes, cujson::cpu::Mode::Lines).expect("cpu tape");
        cpu.extend(d.lines().map(walk::walk_cujson));
        match cujson::parse_lines(&b.bytes, cujson::LinesOptions::default()) {
            Ok(d) => gpu.extend(d.lines().map(walk::walk_cujson)),
            Err(e) => gpu_err = Some(e.to_string()),
        }
    }
    sets.push(("simd-json", simd));
    sets.push(("cujson cpu-reference tape", cpu));
    match gpu_err {
        None => sets.push(("cujson GPU tape", gpu)),
        Some(e) => eprintln!("GPU engine skipped: {e}"),
    }
    let mut bad = false;
    for (name, h) in &sets[1..] {
        let mism: Vec<usize> = (0..h.len().max(sets[0].1.len()))
            .filter(|&i| h.get(i) != sets[0].1.get(i))
            .collect();
        println!(
            "{name}: {} rows, {} vs simd-json: {} mismatching",
            h.len(),
            sets[0].1.len(),
            mism.len()
        );
        bad |= !mism.is_empty();
        if let Some(i) = mism.first() {
            println!("  first mismatch at row {i}");
        }
    }
    if bad {
        std::process::exit(1);
    }
    println!(
        "OK: {} rows agree across {} engines",
        sets[0].1.len(),
        sets.len()
    );
}

fn leak(input: &Input, engine: Engine, iters: usize) {
    let corpus = load(input, None);
    let batch = &corpus.batches[0];
    let mb = |kb: u64| kb / 1024;
    println!("iter  rss_mb  hwm_mb  gpu_free_mb");
    for i in 0..iters {
        if let Err(e) = run_batch(engine, Level::Parse, Tape::Gpu, batch) {
            eprintln!("{e}");
            std::process::exit(1);
        }
        let free = cujson::device_memory().map_or(0, |(f, _)| f as u64 >> 20);
        println!(
            "{i:>4}  {:>6}  {:>6}  {free:>11}",
            mb(proc_kb("VmRSS:").unwrap_or(0)),
            mb(proc_kb("VmHWM:").unwrap_or(0)),
        );
    }
}
