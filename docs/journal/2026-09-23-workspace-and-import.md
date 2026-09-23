# 2026-09-23: Workspace and Upstream Import (task 01)

## Current State

- Root Cargo.toml declares a resolver-3 workspace over `crates/*` with shared package metadata (edition 2024, rust-version 1.85, MIT) and workspace dependencies `cujson-sys`, `cujson`, `cudaforge = "=0.1.6"`, `serde_json` (Cargo.toml)
- rust-toolchain.toml pins the stable channel with rustfmt and clippy
- `cujson-sys` declares `links = "cujson"` and a `cuda` feature enabling the optional `cudaforge` build-dependency; its build.rs only prints `rerun-if-changed=build.rs` (crates/cujson-sys/Cargo.toml, crates/cujson-sys/build.rs)
- `cujson` declares features `cuda` (→ `cujson-sys/cuda`), `cpu-reference` (empty) and `serde` (→ optional `serde_json`); lib.rs holds only a doc comment (crates/cujson/Cargo.toml, crates/cujson/src/lib.rs)
- `cujson-cli` builds a binary named `cujson` whose `main` does nothing, with a `cuda` feature forwarding to `cujson/cuda` (crates/cujson-cli/Cargo.toml, crates/cujson-cli/src/main.rs)
- Commit 111b782 imports `cujson/` from upstream AutomataLab/cuJSON `38d27b6e6c4eb74205cf59f4123b0983034405e2` into crates/cujson-sys/cuda/upstream/ — `diff -r` against vendor/cuJSON/cujson reported no differences, and vendor/cuJSON matched a fresh upstream clone at `38d27b6` byte for byte
- Commit 64a9680 adds upstream LICENSE and UPSTREAM.md (repo, commit, patch-commit rule) to crates/cujson-sys/cuda/upstream/, plus both upstream twitter sample datasets under tests/fixtures/ with a provenance README; both fixtures parse with Python's `json` module
- `cargo build --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` and `cargo fmt --check` pass without CUDA (tier 1); `cargo build -p cujson-sys --features cuda` resolves and builds `cudaforge` (tier 2, no kernels compiled yet)

### Container toolchain (outside the repository)

- `/workspace/.cuda/12.8` contains nvcc 12.8.93, cudart 12.8.90 (including `libcudart_static.a`) and cccl 12.8.90 unpacked from NVIDIA redist tarballs listed in `redistrib_12.8.1.json`, with a `lib64 → lib` symlink because nvcc.profile links from `lib64`
- The pip wheel `nvidia-cuda-nvcc-cu12==12.8.*` contains `ptxas` and nvvm libraries but no `nvcc` driver binary, and `nvidia-cuda-runtime-cu12` contains no `libcudart_static.a` — neither can build cuJSON
- `/workspace/.venvs/cuda12` holds a uv-created Python 3.11 venv with maturin and pytest
- `/workspace/.cuda/env.sh` sets RUSTUP_HOME, CARGO_HOME, CUDA_HOME and PATH, and overrides `/workspace/config.toml`'s clang + mold linker settings with `cc` and `-fuse-ld=bfd` — clang and mold are not installed in the container
- Upstream `main.cu` compiles and links with `nvcc -O3 -std=c++17 -arch=sm_80` from `/workspace/.cuda/12.8`, and `ldd` on the resulting binary lists only libstdc++, libgcc_s, libc and libm (tier 2; the binary was not run)

## Missing

- No kernels compile from the workspace; the tasks in docs/plan/02–10 are not started

## Divergence

- README.md describes cuJSON-rs as a "Rust implementation of cuJSON" but the Rust crates contain no parsing code
