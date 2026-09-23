//! `cujson verify`: GPU parse vs CPU reference tape (`cujson::tape::diff_tapes`)
//! and GPU `to_value()` vs `serde_json`, over one file or the bundled
//! fixtures. Exit codes: 0 all PASS, 1 any FAIL, 2 CUDA unavailable
//! (not compiled in, no driver, or no device).

use std::path::PathBuf;
use std::process::ExitCode;

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
        Ok(sv) if sv == gpu_value => {
            checks.record(&format!("{label}: to_value vs serde_json"), true, None)
        }
        Ok(_) => checks.record(
            &format!("{label}: to_value vs serde_json"),
            false,
            Some("root values differ"),
        ),
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
                    Ok(sv) if sv == gv => {}
                    Ok(_) => {
                        checks.record(
                            &format!("{label}: line {idx} to_value vs serde_json"),
                            false,
                            Some("values differ"),
                        );
                        return;
                    }
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
