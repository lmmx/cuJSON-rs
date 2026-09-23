# 2026-09-23: Lane A - cujson-sys (tasks 02, 03, 04)

## Current State

- `crates/cujson-sys/cuda/upstream/` no longer includes `<x86intrin.h>` in utils.cu,
  query/query_iterator.cpp or query/query_iterator_standard_json.cpp; nvcc -c compiles
  each file clean (tier 2, no GPU)
- Every `exit(` call reachable from the sys crate's build is gone: `cujson_error.h`
  defines `struct cujson_error { int code; }` and a `cujson_err` namespace of
  `constexpr int UTF8=1, UNBALANCED=2, INTERNAL=5, EMPTY_INPUT=6` (not a plain enum -
  see Divergence). parse_standard_json.cu's `Parser()` pairError site and
  parse_standard_json()'s UTF8 site, parse_json_lines.cu's `stage3_parser()` pairError
  site and parse_json_lines()'s per-chunk UTF8 site, and both query_iterator files'
  getKey()/getValue() wrong-node sites now `throw cujson_error{cujson_err::*}` instead
  of calling exit(). Each throw site frees the CUDA allocations live in its own stack
  frame first (parse_standard_json.cu:1572-1579,1633-1636,1650-1659; parse_json_lines.cu
  wraps stage3_parser's throw-through and the UTF8 branch to free `res_buf_arrays[0..i)`
  from earlier loop iterations, parse_json_lines.cu:1181-1189,1202-1213). `utils.cu`'s
  `static void checkCuda(cudaError_t)` still calls exit(1) directly and has no callers
  anywhere in the patched tree (grepped) - task 02's brief marks per-call CUDA error
  checking out of scope (task 03 checks once at the ABI boundary instead)
- A follow-up patch (commit "upstream: free tokenizer index buffers on all paths") fixes
  a device-memory leak found in review: `Parser()` (parse_standard_json.cu) and
  `stage3_parser()` (parse_json_lines.cu) never freed `oc_idx` (Tokenize/stage2_tokenizer's
  open_close index buffer) on either the success or the pairError-throw path, and never
  freed `parsed_oc` (the tokens index buffer, later returned as `result_GPU`) on the
  throw path. `parse_standard_json()` additionally never freed `result_GPU` itself after
  copying it to `res_buff` on the success path (parse_json_lines.cu already did this per
  chunk, at :1243). Both files now free `oc_idx` on every path and `parsed_oc` on the
  throw path inside Parser/stage3_parser, and parse_standard_json.cu frees `result_GPU`
  right after its two D2H `cudaMemcpy` calls. The commit body carries the full
  allocation-by-allocation audit table. This can only be verified by source reading and
  compilation here (tier 2) - the no-growth check (repeated parses not growing device
  memory) is task 10's GPU runbook
- No stdout/stderr output remains in `parse_standard_json.cu`, `parse_json_lines.cu`,
  `load_file.cu` or `utils.cu` except `checkCuda()`'s fprintf (same out-of-scope
  reasoning). `utils.cu` no longer defines 17 debug-dump functions
  (printGpuMemoryUsage, print_d32, print8_d, etc.) - all were unreachable (every call
  site was already commented out upstream), confirmed by grep before deletion.
  `query/query_iterator.cpp` and `query/query_iterator_standard_json.cpp` keep their
  own cerr/cout in file-loading code - not named in task 02 patch 3's file list and not
  compiled into either capi_*.cu unity TU
- `parse_standard_json()` no longer calls `cudaFree(input.data)` (was invalid - that
  pointer is host memory) and `parse_json_lines()` no longer calls
  `cudaFreeHost(input.data)` (was freeing the caller's own buffer); input.data is
  read-only to both parsers now (crates/cujson-sys/cuda/upstream/parse_standard_json.cu,
  parse_json_lines.cu)
- `parse_standard_json.cu`'s and `parse_json_lines.cu`'s entire bodies (after their
  local #includes) are wrapped in `namespace cujson_std { ... }` and
  `namespace cujson_lines { ... }` respectively, with matching namespace wraps in
  their .h declarations - this is what lets capi_standard.cu and capi_lines.cu link
  into one archive without duplicate-symbol errors on the kernels/helpers they both
  define (checkAscii, bitMapCreator, count_set_bits, etc., confirmed by an
  `nm -C --defined-only` diff before the fix). `utils.cu` and `load_file.cu` are
  `#include`d raw into both unity headers, so their colliding free functions
  (checkCuda, count_ones_cub, reduce_cub_int, inclusive_scan_inplace_cub, scatter_cub,
  loadJSON, loadJSONLines_chunkCount/chunkSizeBytes/chunkSizeMegaBytes) are `static`
  instead - `load_file.h`'s prototypes for the loadJSONLines_* functions are removed
  (nothing outside load_file.cu called them, and an extern declaration would conflict
  with a static definition in the same TU). `parse_json_lines.h` has its own
  `JSON_LINES_PARSE_H` include guard instead of reusing `STANDARD_PARSE_H`
- `cuJSONResult::fileSize` (cujson_types.h) is documented as a structural token count
  (result_size+2 or lastStructuralIndex+2), not a byte count - field unchanged, only
  the comment
- `crates/cujson-sys/cuda/cujson_capi.h` is the plain-C ABI: `cujson_status` enum
  (OK/UTF8/UNBALANCED/INPUT_TOO_LARGE/CUDA/INTERNAL/EMPTY_INPUT), `cujson_tape` struct
  (structural, pair_pos, len, cuda_error, opaque `_alloc` - no `depth` field, see
  Divergence), and
  cujson_parse_standard/cujson_parse_lines/cujson_tape_free/cujson_status_str/
  cujson_cuda_runtime_version/cujson_device_count/cujson_device_name/
  cujson_compiled_archs. `cc -fsyntax-only -x c` passes (tier 1)
- `capi_standard.cu`/`capi_lines.cu` are one unity TU each (`#include upstream/cujson.h`
  resp. `upstream/cujsonlines.h`); each entry point rejects null out/null-or-zero
  input/sizes >= INT32_MAX-8 before any allocation, catches `cujson_error` (mapping
  UTF8/UNBALANCED to the matching status, everything else including `catch(...)` to
  INTERNAL), and after a successful parse checks `cudaGetLastError()` then
  `cudaDeviceSynchronize()`, freeing the result and returning CUJSON_ERR_CUDA on
  failure. `cujson_parse_lines` builds `cuJSONLinesInput` chunks itself
  (`build_lines_chunks`, capi_lines.cu) by pointing into the caller's buffer, mirroring
  `loadJSONLines_chunkSizeBytes` without a file or a second host allocation.
  `capi_common.cu` implements the version/device/status-string/free functions
- `crates/cujson-sys/build.rs`: unchanged (rerun-if-changed only) without the `cuda`
  feature. With it: resolves an arch list (CUJSON_CUDA_ARCHS override, else
  CUDA12_DEFAULT=75,80,86,89,90+ptx90 or CUDA13_DEFAULT adding 100,120+ptx120 by
  detected nvcc major version), builds every `-gencode` string by hand (never through
  cudaforge's GpuArch, which auto-appends an 'a'/'f' suffix to any numeric cap >= 90
  with no public opt-out), sets `.compute_cap_arch("75")` only so cudaforge's own
  automatic per-file default gencode has a valid target and never calls nvidia-smi,
  passes `-DCUJSON_COMPILED_ARCHS="<list>"`, builds `libcujson.a` via
  `KernelBuilder::build_lib`, links `static=cujson`, `static=cudart_static`,
  `dylib={stdc++,rt,dl,pthread}`, and emits `cargo:archs=<list>`
- `crates/cujson-sys/src/lib.rs` is `#![no_std]`: `cujson_status` as plain `i32`
  consts (not a Rust enum), `#[repr(C)] cujson_tape`, and the `unsafe extern "C"`
  block gated behind `cfg(feature = "cuda")` - the types/consts remain available
  without the feature
- `crates/cujson-sys/tests/layout.rs` (tier 1, uses the host `cc`, not nvcc): compiles
  a scratch C probe against cujson_capi.h printing `sizeof`/`_Alignof`/`offsetof` for
  every `cujson_tape` field and all seven `CUJSON_*` values, and asserts they match
  `core::mem::size_of`/`align_of`/`offset_of!` and the Rust consts. Passes.

## Verification tiers reached

- Tier 1 (no CUDA): `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D
  warnings`, `cargo build --workspace`, `cargo test --workspace` - all pass, including
  the layout test
- Tier 2 (compile+link with CUDA, no GPU): `cargo build/clippy(-D warnings)/test -p
  cujson-sys --features cuda` all pass. `nm` on the built `libcujson.a` shows all 8
  `cujson_*` symbols. Fat-binary arch coverage checked without cuobjdump (absent from
  this toolkit install): `strings` on each object shows `.target sm_90` (plain, no 'a')
  for the embedded PTX; `objcopy -O binary --only-section=.nv_fatbin` plus scanning for
  ELF magic plus `readelf -h` on each extracted sub-image shows 5 SASS cubins per
  object with Flags 0x4b/0x50/0x56/0x59/0x5a = 75/80/86/89/90 in every one of
  capi_standard.o, capi_lines.o, capi_common.o. `CUJSON_CUDA_ARCHS=80` produces
  `cargo:archs=80`. A scratch C program linked against the real built `libcujson.a`
  plus `-lcudart_static -lstdc++ -lrt -ldl -lpthread` (gcc, not nvcc) links clean and
  `ldd` shows no `libcudart`; not executed
- Tier 3 (GPU): nothing - no GPU in this container. Every claim about a kernel
  actually running, a parse producing correct output, or `cudaGetDeviceCount`/
  `cujson_device_name` returning sane values on real hardware is unverified

## Stubbed

- `cujson-sys` exports raw FFI only; nothing in `crates/cujson/` calls it yet (task 06)

## Missing

- No `cujson_tape` field carries a parse's max depth - `cuJSONResult::depth`
  (cujson_types.h) is never assigned by either upstream parser (confirmed by grep, no
  `.depth =` or `->depth =` anywhere in parse_standard_json.cu or parse_json_lines.cu),
  so the field was removed from the shim rather than exposing an indeterminate int
  (see Divergence). A depth, if needed, is computed host-side from the tape
- No test in this lane calls a `cujson_*` FFI function - `tests/layout.rs` only checks
  layout via a separate `cc` compile. The first real call happens in task 06 or task
  10's GPU runbook

## Divergence

- `docs/plan/03-capi-shim.md`'s example header uses a plain `cujson_status` enum with
  `CUJSON_ERR_*` names; `cujson_capi.h` matches that. But `cujson_error.h` (task 02)
  could not reuse those same names for its internal codes - both headers land in the
  same translation unit in every capi_*.cu, and C++ plain enums put their enumerators
  in the enclosing (global) scope, so `CUJSON_ERR_UTF8` from cujson_error.h collided
  with `CUJSON_ERR_UTF8` from cujson_capi.h. Renamed cujson_error.h's codes to a
  `cujson_err` namespace of `constexpr int` (UTF8/UNBALANCED/INTERNAL/EMPTY_INPUT);
  the 7 throw sites across parse_standard_json.cu, parse_json_lines.cu and the two
  query_iterator files use `cujson_err::*` instead of the brief's `CUJSON_ERR_*`
  spelling. Values are unchanged and still line up numerically with `cujson_status`
- `docs/plan/04-sys-crate-build.md` says to set `.compute_cap_arch(first_arch)` (the
  first arch in the resolved list); build.rs always uses the fixed string `"75"`
  instead, specifically to avoid the same auto-suffix problem the brief warns about -
  if `CUJSON_CUDA_ARCHS` ever resolved to a list starting with an arch >= 90,
  `first_arch` would hit it. See task 04's commit message for the full reasoning
- `docs/plan/03-capi-shim.md` doesn't specify `chunk_bytes == 0` behavior for
  `cujson_parse_lines`; this shim treats it as "one chunk for the whole input" rather
  than an error
- `docs/plan/03-capi-shim.md`'s proposed `cujson_tape` includes a `depth` field; removed
  after review found upstream never writes `cuJSONResult::depth`, so the shim would
  have been copying an indeterminate int. Not present in `cujson_capi.h`, src/lib.rs,
  capi_standard.cu/capi_lines.cu/capi_common.cu or tests/layout.rs
