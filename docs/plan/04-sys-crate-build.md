# 04 — `cujson-sys`: build.rs and raw FFI

Depends on: 03. Lane A. Verification tier: 1 for the non-`cuda` build and the Rust FFI layout test; tier 2 for the `cuda` build.

Reference implementation: `/mnt/mistral.rs/mistralrs-paged-attn/build.rs` and its `Cargo.toml` feature wiring. cudaforge source: `/workspace/registry/src/index.crates.io-1949cf8c6b5b557f/cudaforge-0.1.6/`.

## build.rs

- Pick the `fn main` with `#[cfg(feature = "cuda")]` / `#[cfg(not(feature = "cuda"))]`. Without the feature it prints `cargo:rerun-if-changed=build.rs` and returns
- With `cuda`:
  - `cudaforge::KernelBuilder::new().source_files(["cuda/capi_standard.cu", "cuda/capi_lines.cu", "cuda/capi_common.cu"]).watch(["cuda"])` with args `-std=c++17 -O3 -w -Xcompiler -fPIC`
  - **Arch list.** cudaforge emits exactly one `-gencode` per file, from `compute_cap`/`CUDA_COMPUTE_CAP`/`nvidia-smi` (see `builder.rs` ≈405-420 and `compute_cap.rs`). Set `.compute_cap_arch(first_arch)` explicitly so cudaforge never runs `nvidia-smi`, then add one `-gencode=arch=compute_XX,code=sm_XX` per remaining arch, plus `-gencode=arch=compute_YY,code=compute_YY` for the PTX fallback. Watch cudaforge's auto-suffix: numeric caps ≥ 90 become `sm_90a`, whose code does **not** run on later GPUs. Use plain `90`, `100` and `120` through the string API (`compute_cap_arch("90")` — check whether cudaforge's parser keeps a suffix-less string suffix-less, and if not, pass an arch below 90 as the base and put every ≥90 arch in the extra `-gencode` args)
  - Pick the default arch list by nvcc major version: parse `nvcc --version`, or use cudaforge's toolkit detection if it exposes the version. `CUJSON_CUDA_ARCHS` (comma list, `ptx` suffix allowed, e.g. `89` or `80,90,ptx90`) overrides it. Emit `cargo:rerun-if-env-changed=CUJSON_CUDA_ARCHS`
  - Pass the resolved list to the C side as `-DCUJSON_COMPILED_ARCHS="\"...\""` for `cujson_compiled_archs()`
  - `build_lib(out_dir.join("libcujson.a"))`
  - Link: `rustc-link-search=native=<cuda root>/lib64` (also `lib` / `targets/x86_64-linux/lib` if present), `rustc-link-lib=static=cujson`, `rustc-link-lib=static=cudart_static`, `dylib=stdc++`, `dylib=rt`, `dylib=dl`, `dylib=pthread`
  - Clear error text for a missing nvcc: name `NVCC`/`CUDA_HOME`, and say the `cuda` feature is what needs it
- `links = "cujson"` in Cargo.toml; emit `cargo:archs=<list>` metadata so dependents can read `DEP_CUJSON_ARCHS`

## src/lib.rs

- `#![no_std]`-compatible raw declarations mirroring `cujson_capi.h` exactly: `#[repr(C)]` structs, `#[repr(C)]` status enum as `i32` constants (avoid Rust enums for values coming from C), `unsafe extern "C"` block
- Declarations exist only under `#[cfg(feature = "cuda")]`. Without it the crate exports the types and constants but no functions
- Layout test (tier 1): `size_of`/`align_of`/field offsets of `cujson_tape` match a C compile of the header. Easiest route is a small `cc` dev-build that prints offsets from `offsetof`, or `bindgen` run once as a test comparing against the hand-written declarations. Keep bindgen out of the normal build

## Acceptance

- Tier 1: `cargo build -p cujson-sys`, `cargo test -p cujson-sys` (layout test), and clippy all pass without nvcc
- Tier 2 (via task 09, or here if nvcc gets installed): `cargo build -p cujson-sys --features cuda`, and `nm` on the archive shows the `cujson_*` symbols. `ldd` on a test binary shows no `libcudart`
- Journal entry records the tier reached for each claim
