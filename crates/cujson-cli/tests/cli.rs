//! assert_cmd tests for the `cujson` CLI (docs/plan/07-cli.md acceptance).
//! GPU-runtime behavior is not exercised here — those cases are marked
//! `#[ignore = "requires GPU"]`.

use assert_cmd::Command;
use predicates::prelude::*;

fn cli() -> Command {
    Command::cargo_bin("cujson").unwrap()
}

#[test]
fn no_subcommand_is_an_argument_error() {
    cli().assert().failure();
}

#[test]
fn parse_missing_file_argument_is_an_error() {
    cli().arg("parse").assert().failure();
}

#[test]
fn tape_missing_file_is_an_error() {
    cli().arg("tape").assert().failure();
}

#[test]
fn tape_nonexistent_file_is_an_error() {
    cli()
        .args(["tape", "--cpu", "/nonexistent/path/does-not-exist.json"])
        .assert()
        .failure();
}

#[test]
fn tape_cpu_output_shape_on_tiny_input() {
    let dir = tempfile_dir();
    let path = dir.join("tiny.json");
    std::fs::write(&path, b"{\"a\":1}").unwrap();

    cli()
        .args(["tape", "--cpu", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(
            predicate::str::starts_with("idx\toffset\tchar\tpair\n")
                .and(predicate::str::contains("{"))
                .and(predicate::function(|s: &str| s.lines().count() > 1)),
        );
}

#[test]
fn tape_cpu_lines_mode_on_tiny_input() {
    let dir = tempfile_dir();
    let path = dir.join("tiny.jsonl");
    std::fs::write(&path, b"{\"a\":1}\n{\"b\":2}\n").unwrap();

    cli()
        .args(["tape", "--cpu", "--lines", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::starts_with("idx\toffset\tchar\tpair\n"));
}

#[test]
fn tape_cpu_first_and_last_rows_are_brackets() {
    // FORMAT.md §2 / Document::get_char: idx 0 and idx len-1 are the
    // artificial wrapper entries, read as `[`/`]`, never `?`.
    let dir = tempfile_dir();
    let path = dir.join("brackets.json");
    std::fs::write(&path, b"{\"a\":1}").unwrap();

    let assert = cli()
        .args(["tape", "--cpu", path.to_str().unwrap()])
        .assert()
        .success();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).unwrap();
    let mut lines = stdout.lines();
    lines.next(); // header
    let first = lines.next().unwrap();
    let last = stdout.lines().last().unwrap();
    assert_eq!(first.split('\t').nth(2), Some("["), "first row: {first}");
    assert_eq!(last.split('\t').nth(2), Some("]"), "last row: {last}");
}

#[cfg(unix)]
#[test]
fn tape_broken_pipe_on_stdout_exits_cleanly() {
    // `cujson tape FILE | head` must not print "Broken pipe" and exit
    // non-zero — stdout closing early is a normal exit for a streaming dump.
    let output = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "{} tape --cpu tests/fixtures/twitter_sample_small_records.json --lines | head -n 3",
            assert_cmd::cargo::cargo_bin("cujson").to_str().unwrap()
        ))
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(!stderr.contains("Broken pipe"), "stderr: {stderr}");
}

fn tempfile_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("cujson-cli-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// --- no-CUDA-feature behaviour: only meaningful when this test binary is
// itself built without the `cuda` feature (the default `cargo test` run).

#[cfg(not(feature = "cuda"))]
#[test]
fn info_without_cuda_feature_gives_clean_message_and_exit_2() {
    cli()
        .arg("info")
        .assert()
        .code(2)
        .stdout(predicate::str::contains("built without CUDA"));
}

#[cfg(not(feature = "cuda"))]
#[test]
fn verify_without_cuda_feature_gives_clean_message_and_exit_2() {
    cli()
        .arg("verify")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("cuda"));
}

// --- built with `cuda` on a host where CUDA is unusable: still a clean
// error, not a panic, distinct from a plain FAIL. Skipped when a GPU is usable.

#[cfg(feature = "cuda")]
fn gpu_usable() -> bool {
    let usable = cujson::cuda_info().is_ok();
    if usable {
        eprintln!("skipped: a usable GPU is present");
    }
    usable
}

#[cfg(feature = "cuda")]
#[test]
fn info_with_cuda_feature_but_no_driver_errors_cleanly() {
    if gpu_usable() {
        return;
    }
    let assert = cli().arg("info").assert().failure();
    let output = assert.get_output();
    assert!(
        !output.stderr.is_empty(),
        "expected an error message on stderr"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn verify_with_cuda_feature_but_no_driver_gives_exit_2() {
    if gpu_usable() {
        return;
    }
    cli()
        .arg("verify")
        .assert()
        .code(2)
        .stderr(predicate::str::contains("CUDA"));
}

// --- GPU-dependent behaviour, not runnable in this container.

#[test]
#[ignore = "requires GPU"]
fn verify_bundled_fixtures_passes_on_a_real_gpu() {
    cli().arg("verify").assert().success();
}

#[test]
#[ignore = "requires GPU"]
fn parse_prints_summary_on_a_real_gpu() {
    cli()
        .args(["parse", "tests/fixtures/twitter_sample_large_record.json"])
        .assert()
        .success();
}
