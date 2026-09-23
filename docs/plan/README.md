# cuJSON-rs work plan

This directory holds forward-looking task briefs. `docs/journal/` records only what
exists (see `docs/JOURNAL.md`), so plans live here and each finished task produces a
journal entry describing the resulting state.

Background: `docs/journal/2026-09-23-mistralrs-packaging-survey.md`.

## Goal

`cargo add cujson --features cuda` and `pip install cujson-cu12` should give a working
GPU JSON parser with no manual nvcc invocation, no per-GPU build, and no CUDA runtime
libraries to install beyond the NVIDIA driver.

## Shared decisions

These apply to every task. A brief that needs to deviate says so explicitly.

| Topic | Decision |
|---|---|
| Layout | Cargo workspace; crates under `crates/`: `cujson-sys`, `cujson`, `cujson-cli` (binary `cujson`), `cujson-py` (Python module `cujson`) |
| Edition / MSRV | edition 2024, `rust-version = "1.85"` |
| Upstream source | cuJSON `cujson/` copied into `crates/cujson-sys/cuda/upstream/` from upstream commit `38d27b6e6c4eb74205cf59f4123b0983034405e2`, with its MIT LICENSE and an `UPSTREAM.md` naming the commit. `vendor/` is a read-only reference and will be deleted — nothing may build from or link to it |
| Upstream patches | Applied to the copied sources as separate commits after a pristine-import commit, so `git diff <import-commit> -- crates/cujson-sys/cuda/upstream` shows every change to upstream |
| Kernel build | `cudaforge` build-dependency behind `cujson-sys`'s `cuda` feature (same pattern as mistralrs-paged-attn). Without `cuda`, `build.rs` does nothing and needs no nvcc |
| GPU architectures | One fat binary per CUDA major, not one build per SM. Default arch list lives in one constant in `build.rs`; `CUJSON_CUDA_ARCHS` env var overrides it (e.g. `CUJSON_CUDA_ARCHS=89` for a fast local build). CUDA 12: SASS for 75,80,86,89,90 plus PTX for 90. CUDA 13: SASS for 75,80,86,89,90,100,120 plus PTX for 120 |
| CUDA runtime | Link `cudart_static`. Shipped artifacts then depend only on the driver (`libcuda.so.1`, loaded at runtime), so no CUDA libs get bundled into wheels or tarballs |
| Cargo features | `cujson/cuda` → `cujson-sys/cuda`. `cujson/cpu-reference` enables the CPU tape builder (task 05). `cujson/serde` enables conversion to `serde_json::Value` |
| No-CUDA behaviour | Builds without `cuda` compile and link; `cujson::parse` returns `Error::CudaNotCompiled` |
| Platform | Linux x86_64 first. Removing `x86intrin.h` (task 02) removes the only x86-only code, so aarch64 stays open for later |
| Input size | cuJSON uses `int` sizes; inputs of 2^31 − 1 − padding bytes or more are rejected with an error before reaching C++ |
| Concurrency | cuJSON uses the default stream and no per-call context; the Rust layer serialises GPU parses behind a process-wide `Mutex` |

## Tasks and ordering

```
01 workspace + pristine import
 ├─ lane A: 02 upstream patches → 03 C ABI shim → 04 sys crate build.rs + FFI
 └─ lane B: 05 tape format spec + CPU reference + navigator
        ↓ (both lanes)
06 safe `cujson` crate  (+ 10 GPU validation runbook)
 ├─ 07 CLI
 ├─ 08 Python bindings
 └─ 09 CI + release
```

Lanes A and B touch disjoint directories (`crates/cujson-sys/` vs `crates/cujson/src/tape/`)
and can run as parallel agents in separate git worktrees. 07, 08 and 09 can run in parallel.

## Verification tiers

This container has no GPU. It does have a CUDA 12.8 compile toolchain (nvcc, static
cudart, Thrust/CUB) unpacked from NVIDIA's redist tarballs into `/workspace/.cuda/12.8`,
plus a uv venv with maturin and pytest. Run `. /workspace/.cuda/env.sh` first; it sets
`RUSTUP_HOME`, `CARGO_HOME`, `CUDA_HOME` and `PATH`. Every brief states which tier its
acceptance criteria reach:

1. **No CUDA:** `cargo fmt --check`, `cargo clippy --workspace -- -D warnings`, `cargo test --workspace` without `cuda`
2. **Compile + link with CUDA (no GPU), available here and in CI:** `cargo build --features cuda`, plus `cargo test --features cuda` for tests that don't touch the GPU. Briefs 02–04 that say "none here" / "tier 2 via task 09" can reach tier 2 locally. Unmodified upstream `main.cu` was compiled and linked this way (nvcc 12.8, sm_80), and the result had no CUDA shared-library dependencies
3. **GPU box (user):** anything that runs a kernel, following task 10's runbook

GPU-dependent tests are `#[ignore = "requires GPU"]` so tier 1 stays green, and run on the
GPU box with `--include-ignored`. A claim about runtime behaviour on the GPU is a
hypothesis until tier 3 confirms it, and journal entries say so.

## Rules for implementing agents

- Read `docs/JOURNAL.md`, this README, and your brief before starting
- Stay inside your brief's scope; note out-of-scope findings in your journal entry's Missing/Divergence sections instead of fixing them
- Commit messages: short imperative summary line, optional short body, no `Co-Authored-By` or other trailers
- Finish by writing `docs/journal/YYYY-MM-DD-<task-slug>.md` in the JOURNAL format, stating which verification tier each claim reached
- Never publish (crates.io, PyPI, GitHub releases) or push
