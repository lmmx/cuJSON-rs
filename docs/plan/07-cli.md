# 07 — `cujson` CLI

Depends on: 06. Verification tier: 1 for argument handling and the `cpu-reference` path; tier 3 for GPU commands.

This is the quickest way to try the library: one binary, prebuilt per CUDA major (task 09).

## Commands (clap derive)

- `cujson info`: cuJSON-rs version, compiled archs, CUDA runtime version, visible devices with name and compute capability. Also prints "driver too old" or "no device" diagnostics in plain words. This is the first thing a user runs after install
- `cujson parse FILE [--lines] [--chunk-mb N] [--pointer /a/0/b]... [--time] [--repeat N]`: parses and prints the value(s) at each pointer as JSON, or with no pointer a one-line summary (bytes, tape length, depth, parse ms). `--repeat` reports min/median over N runs, excluding the first (warm-up / context init)
- `cujson tape FILE [--lines] [--cpu]`: dumps the tape as TSV `idx  offset  char  pair`, for debugging and task 10's diffs
- `cujson verify FILE [--lines]`: GPU tape vs CPU reference tape, then GPU `to_value()` vs `serde_json`. Prints the first differing tape index with context and exits non-zero on mismatch. Needs the `cpu-reference` and `serde` features, which the CLI enables by default
- `cujson bench FILE [--repeat N]`: cuJSON vs `serde_json::from_slice::<Value>` vs `simd-json` (optional feature) wall time. Label clearly that cuJSON's time includes the H2D copy

Features: the `cuda` feature forwards to `cujson/cuda`. A CLI built without it still runs `info` (reporting "built without CUDA"), `tape --cpu`, and nothing else.

## Acceptance

- Tier 1: `cargo run -p cujson-cli -- tape --cpu tests/fixtures/twitter_sample_small_records.json --lines` works, and `assert_cmd` tests cover argument errors and the no-CUDA messages
