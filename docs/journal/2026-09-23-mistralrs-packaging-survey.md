# 2026-09-23: mistral.rs Packaging Survey

Reference checkouts: mistral.rs at `ccd265973` (`/mnt/mistral.rs`), cuJSON at `38d27b6` (upstream HEAD 2026-09-10; `vendor/cuJSON` has no .git, identity checked by diff against a fresh clone)
(`vendor/cuJSON`, gitignored), cudaforge 0.1.6 (cargo registry source).

## Current State

### cuJSON-rs repository

- Repository contains README.md, LICENSE, docs/JOURNAL.md and a gitignored `vendor/cuJSON` checkout — no Cargo.toml, build.rs or Rust sources
- Build container has cargo and rustup (stable, nightly) but no CUDA toolkit (`nvcc` absent, `/usr/local/cuda` absent) and no GPU (`nvidia-smi` absent)

### mistral.rs: CUDA kernel compilation

- Each kernel-owning crate (mistralrs-paged-attn, mistralrs-quant, mistralrs-flash-attn, mistralrs-core) compiles its `.cu` sources in its own `build.rs` through the `cudaforge` build-dependency (mistralrs-paged-attn/build.rs:142-185, mistralrs-core/build.rs:106-158)
- `cudaforge` is an optional build-dependency activated only by the crate's `cuda` feature — `cuda = ["dep:cudaforge", "candle-core/cuda"]` (mistralrs-paged-attn/Cargo.toml:32)
- `build.rs` selects one of three `fn main` bodies by `cfg` — CUDA on unix, Metal, or a no-op main for builds with neither feature (mistralrs-paged-attn/build.rs:94, 259, 267-268)
- The no-op `main` declares the same `rustc-check-cfg` names as the CUDA `main`, so CPU-only builds compile without nvcc present (mistralrs-paged-attn/build.rs:267-273)
- `cudaforge::KernelBuilder` takes a source glob, nvcc flags and an output path, runs nvcc per file with content-hash incremental rebuilds, and archives the objects into a static library via `build_lib` (mistralrs-paged-attn/build.rs:142-185)
- `build.rs` links the static archive and the CUDA runtime with `cargo:rustc-link-lib=mistralrspagedattention` and `cargo:rustc-link-lib=dylib=cudart` (mistralrs-paged-attn/build.rs:242-244)
- `build.rs` reads the detected compute capability via `builder.get_compute_cap()` and emits `cargo:rustc-cfg=has_fp8` / `has_fa3_fp8_paged`, gating Rust code paths on GPU architecture at compile time (mistralrs-paged-attn/build.rs:160-167, 246-254)
- Rust calls the compiled kernels through hand-written `extern "C"` declarations with `#[repr(C)]` parameter structs, passing a `CUstream` from candle's re-exported cudarc (mistralrs-paged-attn/src/cuda/ffi.rs:1-3, 65)
- The Rust `cuda` module compiles only under `#[cfg(all(feature = "cuda", target_family = "unix"))]` (mistralrs-paged-attn/src/lib.rs:9-12)
- `cuda` features chain from the top crates downward — mistralrs-pyo3 → mistralrs-core → mistralrs-paged-attn / mistralrs-quant (mistralrs-pyo3/Cargo.toml:50, mistralrs-core/Cargo.toml:123-130)

### cudaforge: toolkit and architecture detection

- cudaforge locates nvcc from the `NVCC` env var, then `PATH`, then `CUDA_HOME/bin`, then `CUDA_PATH`, then a fixed list of install paths (cudaforge src/toolkit.rs:108-140)
- cudaforge resolves the target SM from the `CUDA_COMPUTE_CAP` env var first, then from `nvidia-smi --query-gpu=compute_cap`; the error message tells Docker builds to set `CUDA_COMPUTE_CAP` (cudaforge src/compute_cap.rs:215-250)
- cudaforge appends the `a` suffix (e.g. `sm_90a`) for numeric compute caps ≥ 90 (cudaforge README.md "Numeric (Auto-Suffix)")
- mistral.rs kernels compile for a single SM per build — one binary targets one compute capability, and `get_compute_cap().unwrap_or(80)` defaults to sm_80 when detection fails (mistralrs-paged-attn/build.rs:160)

### mistral.rs: release and Python packaging

- `.github/workflows/release.yml` builds on version tags, running one `linux-cuda` job per (CUDA toolkit × SM × target triple) matrix entry, 36 entries covering CUDA 12.8–13.3 and SM 80–121 (release.yml:302-355)
- Each `linux-cuda` job runs inside `docker.io/nvidia/cuda:<ver>-cudnn-devel-ubuntu22.04` with `CUDA_COMPUTE_CAP` set from the matrix, so no GPU is needed on the builder (release.yml:356-361)
- CLI tarballs bundle every `libcud*`/`libcublas*`/`libnvrtc*` the binary links, found with `ldd`, and set the binary rpath to `$ORIGIN/lib` with patchelf (release.yml:406-417)
- The CUDA wheel step rewrites mistralrs-pyo3's version to `X.Y.Z+cudaNNN.smNN`, builds with `maturin build --features "cuda …" --auditwheel skip`, then runs `auditwheel repair --exclude libcuda.so.1` so CUDA runtime libs ship in the wheel and the driver library does not (release.yml:455-468)
- CUDA wheels upload to the GitHub release as assets, not to PyPI — PyPI carries only CPU (manylinux, Windows) and macOS Metal wheels, and the PyPI README directs CUDA users to `pip install --find-links` (release.yml:286-292, 869; mistralrs-pyo3/README.md:12-20)
- mistralrs-pyo3 builds a `cdylib` named `mistralrs` through maturin, uses `pyo3` with `abi3-py310` so one wheel per platform serves Python ≥3.10, and sets `publish = false` so only the wheel reaches users (mistralrs-pyo3/Cargo.toml, mistralrs-pyo3/pyproject.toml, Cargo.toml `pyo3` workspace dependency)
- `install.sh` reads SM from `nvidia-smi --query-gpu=compute_cap` and the driver CUDA version from `nvidia-smi`, then downloads the matching `mistralrs-cudaNNN-smNN-<triple>.tar.gz` asset (install.sh:139-171, 534-549)
- `ci_cuda.yaml` runs `cargo clippy` and `cargo test --features cuda` on a self-hosted ARM64 GPU runner, limited to same-repo PRs and manual dispatch (ci_cuda.yaml:8-79)

### cuJSON upstream: build and API surface

- cuJSON builds as a unity translation unit — `cujson/cujson.h` `#include`s `utils.cu`, `load_file.cu`, `parse_standard_json.cu` and `query/query_iterator_standard_json.cpp` directly (vendor/cuJSON/cujson/cujson.h)
- Upstream README builds each entry point with a single `nvcc -O3 -std=c++17 -arch=sm_80 <main>.cu` command, requiring CUDA ≥12.1 (vendor/cuJSON/README.md:65, 81, 105-115)
- cuJSON device code depends on Thrust and CUB, both shipped inside the CUDA toolkit (parse_standard_json.cu / parse_json_lines.cu `#include <thrust/…>`, `<cub/cub.cuh>`)
- cuJSON host code includes `<x86intrin.h>` in utils.cu, query_iterator.cpp and query_iterator_standard_json.cpp — the sources compile only on x86 hosts
- Public entry points are C++ functions returning C++ structs containing `std::vector` fields: `loadJSON`, `loadJSONLines_*` (load_file.h), `parse_standard_json(cuJSONInput)` (parse_standard_json.h), `parse_json_lines(cuJSONLinesInput)` (parse_json_lines.h), with results in `cuJSONResult` (cujson_types.h)
- `cuJSONResult` stores sizes as `int` (`fileSize`, `totalResultSize`, `bufferSize`), capping inputs at 2^31−1 bytes (cujson_types.h:22-33)
- `loadJSON` allocates the input buffer with `cudaHostAlloc` pinned memory, and callers release it with `cudaFreeHost` (load_file.cu:21, main.cu:34)
- `parse_standard_json` calls `exit(0)` on invalid UTF-8 and on unbalanced brackets instead of returning an error (parse_standard_json.cu:1572, 1629)
- `parse_json_lines` calls `exit(0)` on parse errors at two sites (parse_json_lines.cu:1064, 1181), and both query iterators call `exit(0)` once each (query_iterator.cpp:607, query_iterator_standard_json.cpp:557)
- `cuJSONIterator`'s constructor takes a file path and re-reads the JSON file from disk rather than using `cuJSONResult::inputJSON` (query_iterator_standard_json.cpp:81-82)
- cuJSON is MIT-licensed (vendor/cuJSON/LICENSE)

## Missing

- cuJSON-rs has no Cargo workspace, `build.rs`, C ABI shim over cuJSON's C++ entry points, Rust bindings, `cuda` feature, Python crate or CI workflows

## Divergence

- README.md describes cuJSON-rs as a "Rust implementation of cuJSON" but the repository contains no Rust code
