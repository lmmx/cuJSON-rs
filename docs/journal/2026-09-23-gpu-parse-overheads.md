# 2026-09-23: GPU parse overheads (host leak, pinned buffers, allocator, concurrency)

Related entries: `2026-09-23-bench.md`, `2026-09-23-bench-measurements.md`, `2026-09-23-safe-api.md` (the `GPU_LOCK` and `TapeStorage::Pinned` this entry changes), `2026-09-23-lane-a-sys-crate.md`.

Version history: the leak fix (ed01209), the single-chunk copy (ce7ae93), the first pinned cache (63a7ea2) and `trim_pinned_cache` in Python went out in v0.1.2 (`82b81aa`, PR #4 squashed as `e81779d`, PR #5 as `1a24d17`); the commits from 5c2250b onward are on branch `faster-walk`, unmerged when this entry was written.

## Current State

### Host-memory leak in JSON Lines mode (ed01209)

- `parse_json_lines.cu` copies each chunk's result into a pinned buffer `res_buf_arrays[i]` and `mergeChunks` copies all of them into one final pinned buffer; before ed01209 the per-chunk buffers were freed only on the error paths, so every successful parse leaked one tape (79 MiB for the benchmark file)
- `parse_standard_json.cu` makes one pinned allocation that the caller receives and frees, and is unaffected
- Observed on the RTX 3090 host before the fix: `leak --iters 30` RSS rose 79 MB per parse, 351 MB to 2,721 MB, with free device memory constant to within 4 MB; 79 MiB equals 10,355,007 entries times 8 bytes
- Observed after the fix: RSS 351 MB, VmHWM 572 MB and free device memory 22,429 MB on all 30 iterations
- The `+MB RSS` column of `run` read 918, 1,471, 2,024, 2,578, 3,131, 3,684, 4,237 and 4,790 on eight successive cuJSON engine rows before the fix and 458 on all eight after it
- `repeated_lines_parses_do_not_grow_host_memory` in `crates/cujson/tests/gpu.rs` parses a 16 MiB input in 2 MiB chunks 33 times and asserts RSS grows by under 100 MB

### Single-chunk copy (ce7ae93)

- With `chunkCount == 1`, `parse_json_lines.cu` copies `structural` to `resultBuffer + 1` and `pair_pos` to `resultBuffer + N + 2` of one pinned allocation of `2N + 3` entries, the layout `mergeChunks` produced, and skips the merge; the multi-chunk path is unchanged
- `parsed_tree.pair_pos` points at `resultBuffer + N + 1`, so `pair_pos[0]` aliases the last `structural` entry and the pair-position of real entry `i` sits at `pair_pos[i + 1]`

### Pinned tape cache (63a7ea2, then fb4aab3)

- `cujson_pinned_alloc` and `cujson_pinned_free` in `capi_common.cu` wrap `cudaMallocHost`/`cudaFreeHost` for the tape buffers of `parse_standard_json.cu`, the single-chunk path and `mergeChunks`; `cujson_tape_free` and the shim's error paths free through `cujson_pinned_free`
- Freed buffers are kept, up to `g_pinned_cache_limit` buffers (1 by default); a request reuses the smallest cached buffer whose capacity is between 1 and 2 times the request; when the cache is full a larger freed buffer replaces the smallest cached one
- `cujson_pinned_cache_set_limit(n)` resizes the cache, and `cujson::set_max_concurrent_parses(n)` calls it with `n + 2`: `n` tapes being built, one queued, one being read
- `cujson_pinned_cache_trim` (Rust: `cujson::trim_pinned_cache()`, Python: `cujson.trim_pinned_cache()`) frees every cached buffer; a mutex guards the cache because a `Document` can be dropped on any thread
- Per-chunk buffers of the multi-chunk path use plain `cudaMallocHost`
- `pinned_buffer_reuse_keeps_tapes_correct` in `gpu.rs` keeps two documents alive at once across four rounds and a trim, comparing each tape with the CPU reference
- Observed with a one-buffer cache and a pipeline holding up to three tapes: `parse busy` for one GPU thread read 0.056 s against 0.047 s for the same parse alone; with the multi-buffer cache it reads 0.049 s

### Pinned input (`PinnedBuffer`, de94433)

- `crates/cujson/src/pinned.rs` defines `PinnedBuffer` (`new`, `from_slice`, `Deref`, `DerefMut`, `Drop`); with the `cuda` feature it owns a `cujson_host_alloc` allocation (a plain `cudaMallocHost` wrapper in `capi_common.cu`, separate from the tape cache), without it a `Vec<u8>`
- CUDA copies from a pinned pointer at the PCIe rate without a bounce buffer; `parse` and `parse_lines` take `&[u8]` and need no API change to use one
- On the RTX 3090 host `--pinned-input` lowered `gpu parse` from 0.052 s to 0.047 s per 233 MB at 32 MB batches; the host-to-device copy ran at 11.6 GB/s from pinned memory in the profile below

### Thrust temporaries (8da77e4)

- All 13 `thrust::cuda::par` calls in `parse_json_lines.cu` and `parse_standard_json.cu` are `thrust::cuda::par(g_cujson_talloc).on(cudaStreamPerThread)`, with `g_cujson_talloc` a stateless `cujson_async_alloc` (`async_alloc.h`) that allocates with `cudaMallocAsync` and frees with `cudaFreeAsync` on `cudaStreamPerThread`
- Thrust's default temporary-buffer allocator calls `cudaMalloc` and `cudaFree` per algorithm call; the pre-change profile counted 203 `cudaMalloc` and 174 `cudaFree` calls over 29 parses

### Concurrency (78bbcf0)

- `ffi.rs` replaces the process-wide `GPU_LOCK` mutex with `MAX_CONCURRENT` slots (`Mutex<usize>` plus `Condvar`); `cujson::set_max_concurrent_parses(n)` sets the count, default 1 (behaviour of the old mutex)
- `capi_lines.cu` and `capi_standard.cu` end a parse with `cudaStreamSynchronize(0)` instead of `cudaDeviceSynchronize()`; the code compiles with per-thread default streams (the profiled API names end in `_ptds` and `_ptsz`), so stream 0 is the calling thread's stream
- The upstream `.cu` files hold no mutable globals (only `static const` values in device functions), and CUDA's error state read by `cudaGetLastError` is per host thread
- `concurrent_parses_match_cpu_reference` in `gpu.rs` runs 4 threads for 20 rounds of a standard and a JSON Lines parse and compares every tape with the CPU reference

### GPU test changes (5d86bc1)

- Every test in `gpu.rs` holds a process-wide `serial()` guard for its duration, because several sample GPU-wide memory
- `repeated_invalid_parses_do_not_grow_device_memory` makes one valid parse before it samples `nvidia-smi` memory: creating the CUDA context costs about 300 MB of device memory, which fell inside the measured window when the test ran alone (1,321 MB to 1,619 MB) or first (1,333 MB to 1,633 MB), and the test passed when run after other tests

## Profiles (`nsys profile --stats=true`, RTX 3090, PCIe link about 12 GB/s)

- 256 MB batch, pageable input, before ce7ae93 (4 parses of 233 MB and one `{}` probe): per parse about 2 ms of kernels (`bitMapCreatorSimd` 0.55 ms, `extractStructuralIdx` 0.37 ms, `fusedStep3_4` 0.27 ms, `checkAscii` 0.27 ms), host-to-device copy 24.6 ms for 233.4 MB (9.5 GB/s), two `cudaHostAlloc` calls of about 17.5 ms, two `cudaFreeHost` calls of about 6.5 ms, two device-to-host copies of 41.4 MB at about 4 ms
- 32 MB batches, pinned input, after 63a7ea2 (28 timed parses and a probe), per 233 MB pass: host-to-device copy 20.2 ms (11.6 GB/s), device-to-host 8.0 ms, kernels 2.4 ms, `cudaStreamSynchronize` 2.7 ms, `cudaMalloc` plus `cudaFree` 3.8 ms, `cudaMallocAsync` 1.4 ms, `cudaLaunchKernel` 1.1 ms, `cudaDeviceSynchronize` 0.45 ms
- Thrust ranges in the second profile: `inclusive_scan` 87 calls at about 120 us, `stable_sort_by_key` 29 at about 182 us, `exclusive_scan` 29 at about 133 us

## Missing

- A persistent device workspace for the input buffer and the remaining per-parse `cudaMalloc` call
- Skipping the `pair_pos` device-to-host copy (41 MB of 82 MB per 233 MB pass) for consumers that only use `Visitor`
- Overlapping a batch's input copy with the previous batch's tape copy within one call (`parse_lines` is one synchronous call per chunk)
- A Python binding for `set_max_concurrent_parses` and `PinnedBuffer`
