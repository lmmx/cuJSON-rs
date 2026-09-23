use std::hint::black_box;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use simd_json::Buffers;

use crate::data::Batch;
use crate::walk::{walk_cujson, walk_simd};

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
    /// cuJSON GPU parse, one-thread tape walk
    Cujson,
    /// cuJSON GPU parse, rayon tape walk
    CujsonPar,
}

impl Engine {
    pub fn is_gpu(self) -> bool {
        matches!(self, Engine::Cujson | Engine::CujsonPar)
    }
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
    pub rows: u64,
    pub hash: u64,
}

impl Run {
    pub fn total(&self) -> Duration {
        self.phases.iter().map(|p| p.1).sum()
    }
}

fn rows_mut(work: &mut [u8]) -> impl Iterator<Item = &mut [u8]> {
    work.split_mut(|&b| b == b'\n').filter(|r| !r.is_empty())
}

pub fn run_batch(engine: Engine, level: Level, batch: &Batch) -> Result<Run, String> {
    match engine {
        Engine::Simd | Engine::SimdPar | Engine::SimdBuf | Engine::SimdBufPar => {
            Ok(run_simd(engine, level, batch))
        }
        Engine::Cujson | Engine::CujsonPar => run_cujson(engine, level, batch),
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
        phases: vec![
            ("copy", copy),
            (if walk { "parse+walk" } else { "parse" }, main),
        ],
        rows,
        hash,
    }
}

fn run_cujson(engine: Engine, level: Level, batch: &Batch) -> Result<Run, String> {
    let t = Instant::now();
    let doc = cujson::parse_lines(&batch.bytes, cujson::LinesOptions::default())
        .map_err(|e| e.to_string())?;
    let parse = t.elapsed();
    if level == Level::Parse {
        let t = Instant::now();
        drop(black_box(doc));
        return Ok(Run {
            phases: vec![("gpu parse", parse), ("free", t.elapsed())],
            rows: batch.rows as u64,
            hash: 0,
        });
    }
    let t = Instant::now();
    let (rows, hash) = if engine == Engine::CujsonPar {
        let nodes: Vec<_> = doc.lines().collect();
        nodes
            .par_iter()
            .map(|n| (1u64, walk_cujson(*n)))
            .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1.wrapping_add(b.1)))
    } else {
        doc.lines().fold((0u64, 0u64), |a, n| {
            (a.0 + 1, a.1.wrapping_add(walk_cujson(n)))
        })
    };
    let walk = t.elapsed();
    let t = Instant::now();
    drop(black_box(doc));
    Ok(Run {
        phases: vec![("gpu parse", parse), ("walk", walk), ("free", t.elapsed())],
        rows,
        hash,
    })
}
