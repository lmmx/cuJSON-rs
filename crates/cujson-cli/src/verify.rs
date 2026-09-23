//! `cujson verify`: GPU parse vs CPU reference tape (`cujson::tape::diff_tapes`)
//! and GPU `to_value()` vs `serde_json`, over one file or the bundled
//! fixtures. Exit codes: 0 all PASS, 1 any FAIL, 2 CUDA unavailable
//! (not compiled in, no driver, or no device).

use std::path::PathBuf;
use std::process::ExitCode;

/// Longest single value rendered in a mismatch report before truncation.
#[cfg(any(feature = "cuda", test))]
const TRUNCATE_AT: usize = 200;

#[cfg(any(feature = "cuda", test))]
fn truncate(s: &str) -> String {
    if s.len() <= TRUNCATE_AT {
        s.to_string()
    } else {
        // Round down to a char boundary so a multi-byte UTF-8 sequence
        // isn't split.
        let mut end = TRUNCATE_AT;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}... ({} bytes total)", &s[..end], s.len())
    }
}

#[cfg(any(feature = "cuda", test))]
fn escape_pointer_token(tok: &str) -> String {
    tok.replace('~', "~0").replace('/', "~1")
}

/// First JSON-Pointer path (RFC 6901) at which `a` and `b` differ, with
/// both values at that path (each truncated to `TRUNCATE_AT` bytes) — or
/// `None` if they're equal. Recurses into objects/arrays only while both
/// sides agree on shape; any type/key/length mismatch is reported at the
/// pointer to the mismatched container itself. Used by both
/// `verify_standard` (root-vs-root) and `verify_lines` (per line), so a
/// `to_value vs serde_json` FAIL always names where the values disagree
/// instead of just "values differ".
#[cfg(any(feature = "cuda", test))]
fn first_value_diff(a: &serde_json::Value, b: &serde_json::Value) -> Option<(String, String, String)> {
    fn go(pointer: &str, a: &serde_json::Value, b: &serde_json::Value) -> Option<String> {
        use serde_json::Value;
        match (a, b) {
            (Value::Object(am), Value::Object(bm)) => {
                if am.len() != bm.len() || am.keys().any(|k| !bm.contains_key(k)) {
                    return Some(pointer.to_string());
                }
                for (k, av) in am {
                    let bv = bm.get(k)?;
                    let child = format!("{pointer}/{}", escape_pointer_token(k));
                    if let Some(p) = go(&child, av, bv) {
                        return Some(p);
                    }
                }
                None
            }
            (Value::Array(aa), Value::Array(ba)) => {
                if aa.len() != ba.len() {
                    return Some(pointer.to_string());
                }
                for (i, (av, bv)) in aa.iter().zip(ba.iter()).enumerate() {
                    let child = format!("{pointer}/{i}");
                    if let Some(p) = go(&child, av, bv) {
                        return Some(p);
                    }
                }
                None
            }
            _ if a == b => None,
            _ => Some(pointer.to_string()),
        }
    }

    let pointer = go("", a, b)?;
    // Re-resolve the pointer to fetch the (possibly non-scalar) values at
    // the point of disagreement, for the printed report.
    let av = if pointer.is_empty() {
        a
    } else {
        resolve_pointer(a, &pointer).unwrap_or(a)
    };
    let bv = if pointer.is_empty() {
        b
    } else {
        resolve_pointer(b, &pointer).unwrap_or(b)
    };
    let pointer = if pointer.is_empty() {
        "/".to_string()
    } else {
        pointer
    };
    Some((pointer, truncate(&av.to_string()), truncate(&bv.to_string())))
}

#[cfg(any(feature = "cuda", test))]
fn resolve_pointer<'v>(v: &'v serde_json::Value, pointer: &str) -> Option<&'v serde_json::Value> {
    v.pointer(pointer)
}

#[cfg(feature = "cuda")]
const LARGE_RECORD: &[u8] = include_bytes!("../../../tests/fixtures/twitter_sample_large_record.json");
#[cfg(feature = "cuda")]
const SMALL_RECORDS: &[u8] = include_bytes!("../../../tests/fixtures/twitter_sample_small_records.json");

#[cfg(feature = "cuda")]
struct Checks {
    total: usize,
    failed: usize,
}

#[cfg(feature = "cuda")]
impl Checks {
    fn new() -> Self {
        Checks { total: 0, failed: 0 }
    }

    fn record(&mut self, label: &str, ok: bool, detail: Option<&str>) {
        self.total += 1;
        if ok {
            println!("PASS: {label}");
        } else {
            self.failed += 1;
            println!("FAIL: {label}");
            if let Some(d) = detail {
                println!("      {d}");
            }
        }
    }
}

pub fn run(file: Option<PathBuf>, lines: bool, chunk_bytes: Option<usize>) -> ExitCode {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (file, lines, chunk_bytes);
        eprintln!("cujson-cli was built without the `cuda` feature (rebuild with --features cuda)");
        ExitCode::from(2)
    }

    #[cfg(feature = "cuda")]
    {
        let info = match cujson::cuda_info() {
            Ok(i) => i,
            Err(e) => {
                eprintln!("CUDA unavailable: {e}");
                if let Some(hint) = crate::cuda_error_hint(&e) {
                    eprintln!("{hint}");
                }
                return ExitCode::from(2);
            }
        };
        println!("CUDA runtime version: {}", info.runtime_version);
        println!("CUDA driver version: {}", info.driver_version);
        println!("compiled archs: {}", info.compiled_archs);
        if info.devices.is_empty() {
            eprintln!("CUDA unavailable: no devices visible");
            eprintln!(
                "{}",
                crate::cuda_error_hint(&cujson::Error::NoDevice).unwrap_or_default()
            );
            return ExitCode::from(2);
        }
        for d in &info.devices {
            println!("device {}: {}", d.index, d.name);
        }

        let mut checks = Checks::new();

        match file {
            Some(path) => {
                let bytes = match std::fs::read(&path) {
                    Ok(b) => b,
                    Err(e) => {
                        eprintln!("error reading {}: {e}", path.display());
                        return ExitCode::FAILURE;
                    }
                };
                let label = path.display().to_string();
                if lines {
                    verify_lines(&mut checks, &label, &bytes, chunk_bytes.unwrap_or(0));
                } else {
                    verify_standard(&mut checks, &label, &bytes);
                }
            }
            None => {
                verify_standard(&mut checks, "twitter_sample_large_record.json (standard)", LARGE_RECORD);
                verify_lines(
                    &mut checks,
                    "twitter_sample_small_records.json (lines, one chunk)",
                    SMALL_RECORDS,
                    0,
                );
                verify_lines(
                    &mut checks,
                    "twitter_sample_small_records.json (lines, small chunks)",
                    SMALL_RECORDS,
                    4096,
                );
            }
        }

        verify_error_recovery(&mut checks);
        verify_memory_growth(&mut checks);

        println!("---");
        println!(
            "{}/{} checks passed",
            checks.total - checks.failed,
            checks.total
        );
        if checks.failed > 0 {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }
}

#[cfg(feature = "cuda")]
fn verify_standard(checks: &mut Checks, label: &str, bytes: &[u8]) {
    let gpu = match cujson::parse(bytes) {
        Ok(d) => d,
        Err(e) => {
            checks.record(&format!("{label}: GPU parse"), false, Some(&e.to_string()));
            return;
        }
    };
    checks.record(&format!("{label}: GPU parse ok"), true, None);

    let cpu = match cujson::cpu::parse(bytes, cujson::cpu::Mode::Standard) {
        Ok(d) => d,
        Err(e) => {
            checks.record(&format!("{label}: CPU reference parse"), false, Some(&e.to_string()));
            return;
        }
    };

    match cujson::tape::diff_tapes(bytes, &gpu.tape, &cpu.tape) {
        None => checks.record(&format!("{label}: tape vs CPU reference"), true, None),
        Some(diff) => checks.record(
            &format!("{label}: tape vs CPU reference"),
            false,
            Some(&diff.to_string()),
        ),
    }

    let gpu_value = gpu.root().to_value();
    let serde_value: Result<serde_json::Value, _> = serde_json::from_slice(bytes);
    match serde_value {
        Ok(sv) => match first_value_diff(&gpu_value, &sv) {
            None => checks.record(&format!("{label}: to_value vs serde_json"), true, None),
            Some((pointer, a, b)) => checks.record(
                &format!("{label}: to_value vs serde_json"),
                false,
                Some(&format!(
                    "first mismatch at {pointer}: gpu={a} serde_json={b}"
                )),
            ),
        },
        Err(e) => checks.record(
            &format!("{label}: to_value vs serde_json"),
            false,
            Some(&format!("serde_json failed to parse reference: {e}")),
        ),
    }
}

#[cfg(feature = "cuda")]
fn verify_lines(checks: &mut Checks, label: &str, bytes: &[u8], chunk_bytes: usize) {
    let opts = cujson::LinesOptions { chunk_bytes };
    let gpu = match cujson::parse_lines(bytes, opts) {
        Ok(d) => d,
        Err(e) => {
            checks.record(&format!("{label}: GPU parse"), false, Some(&e.to_string()));
            return;
        }
    };
    checks.record(&format!("{label}: GPU parse ok"), true, None);

    let cpu = match cujson::cpu::parse(bytes, cujson::cpu::Mode::Lines) {
        Ok(d) => d,
        Err(e) => {
            checks.record(&format!("{label}: CPU reference parse"), false, Some(&e.to_string()));
            return;
        }
    };

    match cujson::tape::diff_tapes(bytes, &gpu.tape, &cpu.tape) {
        None => checks.record(&format!("{label}: tape vs CPU reference"), true, None),
        Some(diff) => checks.record(
            &format!("{label}: tape vs CPU reference"),
            false,
            Some(&diff.to_string()),
        ),
    }

    let text = String::from_utf8_lossy(bytes);
    let mut serde_lines = text.lines().filter(|l| !l.trim().is_empty());
    let mut gpu_lines = gpu.lines();
    let mut idx = 0usize;
    loop {
        let g = gpu_lines.next();
        let s = serde_lines.next();
        match (g, s) {
            (None, None) => break,
            (Some(gnode), Some(sline)) => {
                let gv = gnode.to_value();
                match serde_json::from_str::<serde_json::Value>(sline) {
                    Ok(sv) => match first_value_diff(&gv, &sv) {
                        None => {}
                        Some((pointer, a, b)) => {
                            checks.record(
                                &format!("{label}: line {idx} to_value vs serde_json"),
                                false,
                                Some(&format!(
                                    "first mismatch at {pointer}: gpu={a} serde_json={b}"
                                )),
                            );
                            return;
                        }
                    },
                    Err(e) => {
                        checks.record(
                            &format!("{label}: line {idx} to_value vs serde_json"),
                            false,
                            Some(&format!("serde_json failed: {e}")),
                        );
                        return;
                    }
                }
                idx += 1;
            }
            _ => {
                checks.record(
                    &format!("{label}: line count vs serde_json"),
                    false,
                    Some("line counts differ"),
                );
                return;
            }
        }
    }
    checks.record(
        &format!("{label}: {idx} lines to_value vs serde_json"),
        true,
        None,
    );
}

/// Confirms cuJSON's error paths clean up correctly — the specific
/// unverified claim task 10's runbook cares about most (lane A's
/// error-recovery fix): an invalid-UTF-8 or unbalanced parse must not
/// wedge the device so a valid parse afterwards fails or produces a bad
/// tape. Tier 3 / unverified beyond this container: real error-path
/// behaviour (as opposed to the "no driver" `Error::Cuda` this container
/// always gets first) has only been exercised on a real GPU by task 06's
/// `crates/cujson/tests/gpu.rs` (`#[ignore = "requires GPU"]` here).
#[cfg(feature = "cuda")]
fn verify_error_recovery(checks: &mut Checks) {
    const INVALID_UTF8: &[u8] = b"{\"a\":\"\xFF\"}";
    const UNBALANCED: &[u8] = b"{\"a\":[1,2}";
    const VALID: &[u8] = b"{\"a\":[1,2,3]}";
    const INVALID_UTF8_LINES: &[u8] = b"{\"a\":\"\xFF\"}\n";
    const UNBALANCED_LINES: &[u8] = b"{\"a\":[1,2}\n";
    const VALID_LINES: &[u8] = b"{\"a\":1}\n{\"b\":2}\n";

    fn expect_err<T: std::fmt::Debug>(
        checks: &mut Checks,
        label: &str,
        result: Result<T, cujson::Error>,
        want: fn(&cujson::Error) -> bool,
    ) {
        match result {
            Err(e) if want(&e) => checks.record(label, true, None),
            Err(e) => checks.record(label, false, Some(&format!("wrong error: {e:?}"))),
            Ok(v) => checks.record(label, false, Some(&format!("expected an error, got {v:?}"))),
        }
    }

    fn expect_valid_after(checks: &mut Checks, label: &str, bytes: &[u8], mode: cujson::cpu::Mode) {
        let gpu = match if mode == cujson::cpu::Mode::Standard {
            cujson::parse(bytes)
        } else {
            cujson::parse_lines(bytes, cujson::LinesOptions::default())
        } {
            Ok(d) => d,
            Err(e) => {
                checks.record(label, false, Some(&format!("GPU parse failed: {e}")));
                return;
            }
        };
        let cpu = match cujson::cpu::parse(bytes, mode) {
            Ok(d) => d,
            Err(e) => {
                checks.record(label, false, Some(&format!("CPU reference parse failed: {e}")));
                return;
            }
        };
        match cujson::tape::diff_tapes(bytes, &gpu.tape, &cpu.tape) {
            None => checks.record(label, true, None),
            Some(diff) => checks.record(label, false, Some(&diff.to_string())),
        }
    }

    expect_err(
        checks,
        "error recovery: standard invalid UTF-8 -> InvalidUtf8",
        cujson::parse(INVALID_UTF8),
        |e| matches!(e, cujson::Error::InvalidUtf8),
    );
    expect_err(
        checks,
        "error recovery: standard unbalanced -> Unbalanced",
        cujson::parse(UNBALANCED),
        |e| matches!(e, cujson::Error::Unbalanced),
    );
    expect_valid_after(
        checks,
        "error recovery: standard valid parse after errors",
        VALID,
        cujson::cpu::Mode::Standard,
    );

    expect_err(
        checks,
        "error recovery: lines invalid UTF-8 -> InvalidUtf8",
        cujson::parse_lines(INVALID_UTF8_LINES, cujson::LinesOptions::default()),
        |e| matches!(e, cujson::Error::InvalidUtf8),
    );
    expect_err(
        checks,
        "error recovery: lines unbalanced -> Unbalanced",
        cujson::parse_lines(UNBALANCED_LINES, cujson::LinesOptions::default()),
        |e| matches!(e, cujson::Error::Unbalanced),
    );
    expect_valid_after(
        checks,
        "error recovery: lines valid parse after errors",
        VALID_LINES,
        cujson::cpu::Mode::Lines,
    );
}

/// Heuristic device-memory-growth check — task 10's other unverified
/// claim (lane A's leak fixes). Runs a warm-up, records free memory,
/// makes 700 parses (300 valid standard, 300 invalid alternating
/// UTF-8/unbalanced, 100 lines-mode multi-chunk), records free memory
/// again. PASS if the drop is under 32 MiB. This is a heuristic, not a
/// leak proof: CUDA's stream-ordered allocator can hold freed memory in
/// its pool rather than returning it to the driver, so a `0` delta here
/// doesn't prove no allocator churn, and a per-parse leak of the size
/// lane A's fix addressed (~0.3-1 MB) would still clear 32 MiB many times
/// over across 700 parses and show clearly. Tier 3 / unverified: this
/// container has no device, so this check has only run against the
/// no-driver `Err` path here, never against a real GPU's memory counters.
#[cfg(feature = "cuda")]
fn verify_memory_growth(checks: &mut Checks) {
    const INVALID_UTF8: &[u8] = b"{\"a\":\"\xFF\"}";
    const UNBALANCED: &[u8] = b"{\"a\":[1,2}";
    const THRESHOLD_BYTES: usize = 32 * 1024 * 1024;

    for _ in 0..5 {
        let _ = cujson::parse(LARGE_RECORD);
    }

    let before = match cujson::device_memory() {
        Ok((free, _total)) => free,
        Err(e) => {
            checks.record(
                "memory growth: device_memory (before)",
                false,
                Some(&e.to_string()),
            );
            return;
        }
    };

    for _ in 0..300 {
        let _ = cujson::parse(LARGE_RECORD);
    }
    for i in 0..300 {
        if i % 2 == 0 {
            let _ = cujson::parse(INVALID_UTF8);
        } else {
            let _ = cujson::parse(UNBALANCED);
        }
    }
    for _ in 0..100 {
        let _ = cujson::parse_lines(SMALL_RECORDS, cujson::LinesOptions { chunk_bytes: 4096 });
    }

    let after = match cujson::device_memory() {
        Ok((free, _total)) => free,
        Err(e) => {
            checks.record(
                "memory growth: device_memory (after)",
                false,
                Some(&e.to_string()),
            );
            return;
        }
    };

    let delta = before.saturating_sub(after); // free shrank => used grew
    println!(
        "memory: before={before} bytes free, after={after} bytes free, delta={delta} bytes \
         (heuristic — CUDA's stream-ordered allocator pool can hold freed memory)"
    );
    let ok = delta < THRESHOLD_BYTES;
    let detail = format!(
        "delta={delta} bytes over 700 parses (threshold {THRESHOLD_BYTES} bytes); heuristic, see above"
    );
    checks.record(
        "memory growth over 700 parses stays under 32 MiB (heuristic)",
        ok,
        if ok { None } else { Some(&detail) },
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn equal_values_have_no_diff() {
        let a = json!({"x": [1, 2, {"y": "z"}]});
        let b = a.clone();
        assert!(first_value_diff(&a, &b).is_none());
    }

    #[test]
    fn top_level_scalar_mismatch_points_at_root() {
        let (pointer, a, b) = first_value_diff(&json!(1), &json!(2)).unwrap();
        assert_eq!(pointer, "/");
        assert_eq!(a, "1");
        assert_eq!(b, "2");
    }

    #[test]
    fn nested_object_mismatch_reports_full_pointer() {
        let a = json!({"a": {"b": [1, 2, 3]}});
        let b = json!({"a": {"b": [1, 9, 3]}});
        let (pointer, av, bv) = first_value_diff(&a, &b).unwrap();
        assert_eq!(pointer, "/a/b/1");
        assert_eq!(av, "2");
        assert_eq!(bv, "9");
    }

    #[test]
    fn missing_key_reports_pointer_to_object() {
        let a = json!({"a": 1, "b": 2});
        let b = json!({"a": 1});
        let (pointer, _, _) = first_value_diff(&a, &b).unwrap();
        assert_eq!(pointer, "/");
    }

    #[test]
    fn pointer_token_escaping() {
        let a = json!({"a/b": 1});
        let b = json!({"a/b": 2});
        let (pointer, _, _) = first_value_diff(&a, &b).unwrap();
        assert_eq!(pointer, "/a~1b");
    }

    #[test]
    fn long_values_are_truncated() {
        let long = "x".repeat(500);
        let out = truncate(&long);
        assert!(out.len() < 500);
        assert!(out.contains("500 bytes total"));
    }

    #[test]
    fn short_values_are_not_truncated() {
        assert_eq!(truncate("short"), "short");
    }
}
