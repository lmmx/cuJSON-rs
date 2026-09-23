# 2026-09-23: Integration Review of Tasks 07 and 08

## Current State

- `packaging` contains tasks 01–09; task 10 (GPU validation) has not been run — no code path has executed on a GPU (tier 3 absent for every GPU-runtime claim)
- `cujson-cli` restores the default SIGPIPE action at the start of `main` via `libc::signal(SIGPIPE, SIG_DFL)` on unix (crates/cujson-cli/src/main.rs) — commit 72b222f replaced a `catch_unwind` on "Broken pipe" panic payloads; `cujson tape --cpu … | head` now exits silently with status 141
- The task 07 branch failed `cargo fmt --check` in crates/cujson-cli and crates/cujson/tests/no_driver_error_mapping.rs; commit 14a7386 applied rustfmt after the merge
- Root Cargo.toml `[workspace.dependencies]` merges task 08's `pyo3 = "0.29"` with task 07's `clap`, `assert_cmd`, `predicates`; Cargo.lock was regenerated from `packaging`'s lockfile by cargo rather than hand-merged (merge commit d39d7e9)
- `cujson.InputError` (CujsonError, ValueError) is re-exported from `cujson/__init__.py` and `__init__.pyi` — task 08 created the class in `_cujson` but omitted it from the package (commit 60a6a27); `test_every_extension_export_is_reexported` asserts every public `_cujson` name appears in `cujson.__all__`
- `cujson.CUDA_COMPILED` is a module-level bool set from `cfg!(feature = "cuda")` (crates/cujson-py/src/lib.rs) — `cuda_info()` raises on a CUDA wheel without a driver, so build-dependent tests branch on `CUDA_COMPILED` instead (commit da5aa80)
- With the CUDA wheel in this driverless container, `cujson.cuda_info()`, `cujson.parse()` and `cujson.parse_lines()` each raise `CudaError` with message "CUDA error 35: cudaErrorInsufficientDriver: …" — before task 07's item 6 fix, `parse`/`parse_lines` raised a generic `CujsonError: internal error` because the shim's `catch (...)` swallowed `thrust::system_error` (tier 2, observed here)
- pytest results in this container: non-CUDA wheel 6 passed / 5 skipped; CUDA wheel (no driver) 5 passed / 6 skipped, including `test_cuda_build_without_gpu_raises_cuda_error`
- Lint and tests on the merged tree pass for workspace feature sets {none, cujson/cpu-reference, cujson/cpu-reference+serde, cujson/cuda, cujson-cli/cuda, cujson-cli/cuda+cujson-py/cuda}, plus `cargo clippy -p cujson-py` with and without `cuda` (CUJSON_CUDA_ARCHS=80)
- `cargo run -p cujson-cli --features cuda -- verify` in this container prints "no NVIDIA driver found" and exits 2
- `cargo run -p cujson-cli --features cuda -- info` on the user's host (RTX 3090, CUDA runtime and driver 13020) built the CUDA 13 default arch list `75,80,86,89,90,100,120;ptx120` in 1m16s (dev profile) and listed device 0 — first tier-3 observation; no parse kernel has run yet

## Missing

- docs/GPU_VALIDATION.md (task 10 runbook) does not exist
