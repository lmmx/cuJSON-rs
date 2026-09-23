//! Tier 3 GPU integration tests — `#[ignore = "requires GPU"]` throughout,
//! so `cargo test --features cuda` (this container, no GPU) stays green.
//! Run on a GPU box with `cargo test --features cuda,cpu-reference,serde -- --include-ignored`
//! (`docs/plan/10-gpu-validation-runbook.md` step 4).
//!
//! Every parse here goes through the public `cujson::parse`/`parse_lines`
//! API, which serializes on `ffi::GPU_LOCK` internally
//! (`crates/cujson/src/ffi.rs`) — no direct `cujson_sys` calls, per
//! `docs/plan/README.md`'s "Concurrency" row.
#![cfg(all(feature = "cuda", feature = "cpu-reference", feature = "serde"))]

use std::path::{Path, PathBuf};

use cujson::cpu;
use cujson::tape::{Mode, diff_tapes};

/// The tests below sample GPU-wide memory, so they must not overlap.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn read_fixture(name: &str) -> Vec<u8> {
    std::fs::read(fixtures_dir().join(name)).expect("fixture file present")
}

#[test]
#[ignore = "requires GPU"]
fn standard_fixture_matches_serde_json() {
    let _serial = serial();
    let bytes = read_fixture("twitter_sample_large_record.json");
    let expected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let doc = cujson::parse(&bytes).expect("GPU parse");
    assert_eq!(doc.root().to_value(), expected);
}

#[test]
#[ignore = "requires GPU"]
fn lines_fixture_matches_serde_json() {
    let _serial = serial();
    let bytes = read_fixture("twitter_sample_small_records.json");
    let expected: Vec<serde_json::Value> = bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();

    let doc =
        cujson::parse_lines(&bytes, cujson::LinesOptions::default()).expect("GPU parse_lines");
    let got: Vec<serde_json::Value> = doc.lines().map(|n| n.to_value()).collect();
    assert_eq!(got, expected);
}

#[test]
#[ignore = "requires GPU"]
fn standard_fixture_tape_matches_cpu_reference() {
    let _serial = serial();
    let bytes = read_fixture("twitter_sample_large_record.json");
    let gpu_doc = cujson::parse(&bytes).expect("GPU parse");
    let cpu_doc = cpu::parse(&bytes, Mode::Standard).expect("cpu::parse");
    let diff = diff_tapes(&bytes, &gpu_doc.tape, &cpu_doc.tape);
    assert!(diff.is_none(), "tape mismatch: {}", diff.unwrap());
}

#[test]
#[ignore = "requires GPU"]
fn lines_fixture_tape_matches_cpu_reference() {
    let _serial = serial();
    let bytes = read_fixture("twitter_sample_small_records.json");
    let gpu_doc =
        cujson::parse_lines(&bytes, cujson::LinesOptions::default()).expect("GPU parse_lines");
    let cpu_doc = cpu::parse(&bytes, Mode::Lines).expect("cpu::parse");
    let diff = diff_tapes(&bytes, &gpu_doc.tape, &cpu_doc.tape);
    assert!(diff.is_none(), "tape mismatch: {}", diff.unwrap());
}

/// A multi-chunk Lines parse (small `chunk_bytes`) is provably equivalent,
/// structural-entry for structural-entry, to a single-chunk parse
/// (`tape/FORMAT.md` §5 case 4) — this is the differential test that
/// actually exercises `cujson_parse_lines`'s chunking path, unlike the
/// fixture test above which likely fits in one chunk.
#[test]
#[ignore = "requires GPU"]
fn lines_multi_chunk_tape_matches_cpu_reference() {
    let _serial = serial();
    let bytes = read_fixture("twitter_sample_small_records.json");
    let small_chunk = cujson::LinesOptions { chunk_bytes: 256 };
    let gpu_doc = cujson::parse_lines(&bytes, small_chunk).expect("GPU parse_lines");
    let cpu_doc = cpu::parse(&bytes, Mode::Lines).expect("cpu::parse");
    let diff = diff_tapes(&bytes, &gpu_doc.tape, &cpu_doc.tape);
    assert!(diff.is_none(), "tape mismatch: {}", diff.unwrap());
}

/// Fixed-seed proptest-style corpus: a few hundred generated JSON values,
/// each round-tripped through the GPU and compared tape-for-tape against
/// the CPU reference builder.
/// A minimal deterministic (fixed-seed) xorshift generator — avoids adding
/// a `rand` dependency just for this one ignored test.
struct Xorshift(u64);
impl Xorshift {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn range(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

fn gen_value(rng: &mut Xorshift, depth: u32) -> serde_json::Value {
    if depth == 0 {
        return serde_json::Value::Null;
    }
    match rng.range(6) {
        0 => serde_json::Value::Null,
        1 => serde_json::Value::Bool(rng.range(2) == 0),
        2 => serde_json::json!((rng.range(2_000_000) as i64) - 1_000_000),
        3 => serde_json::json!(((rng.next_u64() as f64 / u64::MAX as f64) - 0.5) * 1.0e6),
        4 => {
            let n = rng.range(6);
            serde_json::Value::Array((0..n).map(|_| gen_value(rng, depth - 1)).collect())
        }
        _ => {
            let n = rng.range(6);
            let map: serde_json::Map<_, _> = (0..n)
                .map(|i| (format!("k{i}"), gen_value(rng, depth - 1)))
                .collect();
            serde_json::Value::Object(map)
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn fixed_seed_corpus_tape_matches_cpu_reference() {
    let _serial = serial();
    let mut rng = Xorshift(0xC0FFEE_u64);
    for _ in 0..300 {
        let v = gen_value(&mut rng, 5);
        let bytes = serde_json::to_vec(&v).unwrap();
        let gpu_doc = cujson::parse(&bytes).expect("GPU parse");
        let cpu_doc = cpu::parse(&bytes, Mode::Standard).expect("cpu::parse");
        let diff = diff_tapes(&bytes, &gpu_doc.tape, &cpu_doc.tape);
        assert!(diff.is_none(), "tape mismatch on {v:?}: {}", diff.unwrap());
        assert_eq!(gpu_doc.root().to_value(), v);
    }
}

#[test]
#[ignore = "requires GPU"]
fn error_paths_then_valid_parse_still_works() {
    let _serial = serial();
    // Invalid UTF-8.
    let bad_utf8 = [0x7B, 0xFF, 0x7D]; // `{`, invalid byte, `}`
    let err = cujson::parse(&bad_utf8).unwrap_err();
    assert!(matches!(err, cujson::Error::InvalidUtf8));

    // Unbalanced brackets.
    let unbalanced = b"{\"a\":1";
    let err = cujson::parse(unbalanced).unwrap_err();
    assert!(matches!(err, cujson::Error::Unbalanced));

    // A follow-up valid parse must still succeed — checks patch 02-2's
    // per-throw-site cleanup (lane A journal) doesn't leave the next call
    // in a bad state.
    let doc = cujson::parse(b"{\"a\":1}").expect("valid parse after two errors");
    assert_eq!(doc.root().get("a").unwrap().as_i64(), Ok(1));
}

/// Approximate device-memory-growth check: parses fail-then-succeed 1000
/// times, sampling `nvidia-smi --query-gpu=memory.used` before and after.
/// `docs/plan/10-gpu-validation-runbook.md` step 6 is the authoritative
/// version of this check (it also covers valid-parse repetition); this
/// test is a lighter in-process approximation for the same claim.
#[test]
#[ignore = "requires GPU"]
fn repeated_invalid_parses_do_not_grow_device_memory() {
    let _serial = serial();
    fn memory_used_mb() -> Option<u64> {
        let out = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }

    // Creating the CUDA context costs ~300 MB of device memory; take it
    // before sampling.
    let _ = cujson::parse(b"{\"a\":1}");
    let before = memory_used_mb();
    for _ in 0..1000 {
        let _ = cujson::parse(b"{\"a\":1");
    }
    let doc = cujson::parse(b"{\"a\":1}").expect("valid parse after 1000 errors");
    assert_eq!(doc.root().get("a").unwrap().as_i64(), Ok(1));
    let after = memory_used_mb();

    if let (Some(b), Some(a)) = (before, after) {
        // Generous slack: this is an approximation run alongside whatever
        // else shares the GPU, not a precise accounting.
        assert!(
            a <= b + 64,
            "device memory grew by more than 64MB over 1000 invalid parses: {b}MB -> {a}MB"
        );
    }
}

fn rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|l| {
            l.strip_prefix("VmRSS:")?
                .trim()
                .strip_suffix("kB")?
                .trim()
                .parse()
                .ok()
        })
        .unwrap()
}

#[test]
#[ignore = "requires GPU (Linux)"]
fn repeated_lines_parses_do_not_grow_host_memory() {
    let _serial = serial();
    let record = read_fixture("twitter_sample_small_records.json");
    let mut input = Vec::new();
    while input.len() < 16 << 20 {
        input.extend_from_slice(&record);
    }
    let opts = cujson::LinesOptions {
        chunk_bytes: 2 << 20,
    };
    for _ in 0..3 {
        drop(cujson::parse_lines(&input, opts).expect("GPU parse"));
    }
    let before = rss_kb();
    for _ in 0..30 {
        drop(cujson::parse_lines(&input, opts).expect("GPU parse"));
    }
    let grown_mb = rss_kb().saturating_sub(before) / 1024;
    // Each parse's tape is tens of MB; leaking it 30 times is well over this.
    assert!(grown_mb < 100, "RSS grew {grown_mb} MB over 30 parses");
}
