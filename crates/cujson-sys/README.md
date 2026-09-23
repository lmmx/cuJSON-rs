# cujson-sys

Raw FFI bindings to the [cuJSON](https://github.com/AutomataLab/cuJSON) GPU JSON parser. Use the
safe [`cujson`](https://crates.io/crates/cujson) crate instead unless you need the C ABI directly.

The crate bundles cuJSON's CUDA sources (from upstream commit `38d27b6`, patched to be usable as
a library) and a C ABI shim (`cuda/cujson_capi.h`). With the `cuda` feature, its build script
compiles them with `nvcc` via [`cudaforge`](https://crates.io/crates/cudaforge) into a static
library for compute capability 7.5 onwards plus a PTX fallback, and links the CUDA runtime
statically. `CUJSON_CUDA_ARCHS` overrides the architecture list (e.g. `CUJSON_CUDA_ARCHS=86`).
Without the feature nothing is compiled and no functions are declared.

Part of [cuJSON-rs](https://github.com/lmmx/cuJSON-rs). MIT licensed, as is upstream cuJSON.
