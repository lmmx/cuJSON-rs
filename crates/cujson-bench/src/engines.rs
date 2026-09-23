use std::hint::black_box;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use simd_json::Buffers;

use crate::data::Batch;
use crate::walk::{HashVisitor, walk_cujson, walk_simd};

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Engine {
    /// simd-json, one thread, `to_borrowed_value` as genson-rs calls it
    Simd,
    /// simd-json, rayon over rows (how genson-rs runs)
    SimdPar,
    /// simd-json, one thread, reusing parse buffers
    SimdBuf,
    /// simd-json, rayon over rows, per-thread reused buffers
    SimdBufPar,
    /// cuJSON tape, walked with `Node` navigation, one thread
    CujsonNode,
    /// cuJSON tape, `Node` navigation over lines in rayon
    CujsonNodePar,
    /// cuJSON tape, single-pass `Document::visit`
    CujsonVisit,
    /// cuJSON tape, `Document::visit_range` over row ranges in rayon
    CujsonVisitPar,
    /// `cujson-visit-par`, with batch N+1 parsed while batch N is walked (use `--batch-mb` below the file size)
    CujsonPipe,
}

impl Engine {
    pub fn is_cujson(self) -> bool {
        !matches!(
            self,
            Engine::Simd | Engine::SimdPar | Engine::SimdBuf | Engine::SimdBufPar
        )
    }
}

/// Where a cuJSON engine gets its tape.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Tape {
    /// `cujson::parse_lines` (needs the `cuda` feature and a GPU)
    Gpu,
    /// `cujson::cpu::parse`, the sequential reference builder (isolates the walk from the GPU)
    Cpu,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum)]
pub enum Level {
    /// Parse and discard.
    Parse,
    /// Parse, then visit every node (checksum).
    Walk,
}

pub struct Run {
    pub phases: Vec<(&'static str, Duration)>,
    /// Elapsed time when phases overlap; otherwise the phases sum.
    pub wall: Option<Duration>,
    pub rows: u64,
    pub hash: u64,
}

impl Run {
    pub fn total(&self) -> Duration {
        self.wall
            .unwrap_or_else(|| self.phases.iter().map(|p| p.1).sum())
    }
}

fn rows_mut(work: &mut [u8]) -> impl Iterator<Item = &mut [u8]> {
    work.split_mut(|&b| b == b'\n').filter(|r| !r.is_empty())
}

pub fn run_batch(engine: Engine, level: Level, tape: Tape, batch: &Batch) -> Result<Run, String> {
    if engine == Engine::CujsonPipe {
        Err("cujson-pipe runs over all batches".into())
    } else if engine.is_cujson() {
        run_cujson(engine, level, tape, batch)
    } else {
        Ok(run_simd(engine, level, batch))
    }
}

fn run_simd(engine: Engine, level: Level, batch: &Batch) -> Run {
    let par = matches!(engine, Engine::SimdPar | Engine::SimdBufPar);
    let reuse = matches!(engine, Engine::SimdBuf | Engine::SimdBufPar);
    let walk = level == Level::Walk;

    // simd-json rewrites its input in place, so a mutable copy is required.
    let t = Instant::now();
    let mut work = batch.bytes.clone();
    let copy = t.elapsed();

    let one = |row: &mut [u8], bufs: &mut Option<Buffers>| -> (u64, u64) {
        let v = match bufs {
            Some(b) => simd_json::to_borrowed_value_with_buffers(row, b),
            None => simd_json::to_borrowed_value(row),
        }
        .expect("simd-json parse");
        let h = if walk { walk_simd(&v) } else { 0 };
        black_box(&v);
        (1, h)
    };
    let init = || reuse.then(Buffers::default);

    let t = Instant::now();
    let (rows, hash) = if par {
        work.par_split_mut(|&b| b == b'\n')
            .filter(|r| !r.is_empty())
            .map_init(init, |bufs, row| one(row, bufs))
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)))
    } else {
        let mut bufs = init();
        rows_mut(&mut work).fold((0u64, 0u64), |a, row| {
            let b = one(row, &mut bufs);
            (a.0 + b.0, a.1.wrapping_add(b.1))
        })
    };
    let main = t.elapsed();
    Run {
        wall: None,
        phases: vec![
            ("copy", copy),
            (if walk { "parse+walk" } else { "parse" }, main),
        ],
        rows,
        hash,
    }
}

fn run_cujson(engine: Engine, level: Level, tape: Tape, batch: &Batch) -> Result<Run, String> {
    let t = Instant::now();
    let doc = match tape {
        Tape::Cpu => cujson::cpu::parse(batch.input(), cujson::cpu::Mode::Lines)
            .map_err(|e| e.to_string())?,
        Tape::Gpu => cujson::parse_lines(batch.input(), cujson::LinesOptions::default())
            .map_err(|e| e.to_string())?,
    };
    let parse = t.elapsed();
    let parse_name = match tape {
        Tape::Cpu => "cpu tape build",
        Tape::Gpu => "gpu parse",
    };
    if level == Level::Parse {
        let t = Instant::now();
        drop(black_box(doc));
        return Ok(Run {
            wall: None,
            phases: vec![(parse_name, parse), ("free", t.elapsed())],
            rows: batch.rows as u64,
            hash: 0,
        });
    }
    let sum = |a: (u64, u64), b: (u64, u64)| (a.0 + b.0, a.1.wrapping_add(b.1));
    let t = Instant::now();
    let (rows, hash) = match engine {
        Engine::CujsonNode => doc
            .lines()
            .fold((0u64, 0u64), |a, n| sum(a, (1, walk_cujson(n)))),
        Engine::CujsonNodePar => {
            let nodes: Vec<_> = doc.lines().collect();
            nodes
                .par_iter()
                .map(|n| (1u64, walk_cujson(*n)))
                .reduce(|| (0, 0), sum)
        }
        Engine::CujsonVisit => {
            let mut v = HashVisitor::default();
            doc.visit(&mut v).map_err(|e| e.to_string())?;
            (v.rows, v.hash)
        }
        _ => doc
            .split_lines(rayon::current_num_threads() * 16)
            .into_par_iter()
            .map(|r| {
                let mut v = HashVisitor::default();
                doc.visit_range(r, &mut v).expect("visit_range");
                (v.rows, v.hash)
            })
            .reduce(|| (0, 0), sum),
    };
    let walk = t.elapsed();
    let t = Instant::now();
    drop(black_box(doc));
    Ok(Run {
        wall: None,
        phases: vec![(parse_name, parse), ("walk", walk), ("free", t.elapsed())],
        rows,
        hash,
    })
}

fn parse_batch(batch: &Batch, tape: Tape) -> Result<cujson::Document<'_>, String> {
    match tape {
        Tape::Cpu => {
            cujson::cpu::parse(batch.input(), cujson::cpu::Mode::Lines).map_err(|e| e.to_string())
        }
        Tape::Gpu => cujson::parse_lines(batch.input(), cujson::LinesOptions::default())
            .map_err(|e| e.to_string()),
    }
}

/// Parse batch N+1 on one thread while batch N is walked in rayon.
pub fn run_pipeline(batches: &[Batch], tape: Tape) -> Result<Run, String> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let start = Instant::now();
    std::thread::scope(|s| {
        let producer = s.spawn(move || -> Result<Duration, String> {
            let mut busy = Duration::ZERO;
            for b in batches {
                let t = Instant::now();
                let doc = parse_batch(b, tape)?;
                busy += t.elapsed();
                if tx.send(doc).is_err() {
                    break;
                }
            }
            Ok(busy)
        });
        let (mut rows, mut hash, mut walk_busy) = (0u64, 0u64, Duration::ZERO);
        for doc in rx {
            let t = Instant::now();
            let (r, h) = doc
                .split_lines(rayon::current_num_threads() * 16)
                .into_par_iter()
                .map(|range| {
                    let mut v = HashVisitor::default();
                    doc.visit_range(range, &mut v).expect("visit_range");
                    (v.rows, v.hash)
                })
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)));
            rows += r;
            hash = hash.wrapping_add(h);
            drop(doc);
            walk_busy += t.elapsed();
        }
        let parse_busy = producer.join().expect("producer thread")?;
        Ok(Run {
            wall: Some(start.elapsed()),
            phases: vec![("parse busy", parse_busy), ("walk busy", walk_busy)],
            rows,
            hash,
        })
    })
}
