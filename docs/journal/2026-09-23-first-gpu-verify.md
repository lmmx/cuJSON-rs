# 2026-09-23: First GPU Runs of `cujson verify`

Host: NVIDIA GeForce RTX 3090 (sm_86), CUDA runtime 13020, driver 13020, `CUJSON_CUDA_ARCHS=86 cargo gpu-verify` (release build, 31 s).

## Current State

- Standard mode passes all three checks on the large-record fixture — GPU parse, `diff_tapes` against the CPU reference, and `to_value()` against `serde_json` (tier 3) — so FORMAT.md §1–§3 match the kernels' output for this input
- Error recovery passes in both modes: invalid UTF-8 returns `InvalidUtf8`, `{"a":[1,2}` returns `Unbalanced`, and a following valid parse passes the tape diff (tier 3)
- Free device memory reads 23,577,493,504 bytes before and after 700 mixed parses (300 valid, 300 invalid, 100 multi-chunk lines) in the second run, and 23,622,975,488 bytes before and after in the first — delta 0 both times, consistent with the lane A leak fixes (tier 3; heuristic, since the stream-ordered pool can hold memory)
- First run (commit 51a5f33): Lines mode with one chunk failed the tape diff at index 22439 (`structural[len-1]`: GPU 0, CPU 22439), and Lines-mode "valid parse after errors" failed at index 0 (GPU 2119, CPU 0) — `mergeChunks` never writes either artificial wrapper entry (parse_json_lines.cu:1093-1113); commit fdee882 makes the shim write them, and FORMAT.md §5 records the correction
- First run: Lines mode with 4096-byte chunks returned `Error::Internal` — the shim's chunker emitted a zero-size first chunk whenever the first line exceeded `chunk_bytes` (fixture lines average ~6 KB), which parse_json_lines rejects by returning an empty result; commit 10e53f1 moves the chunker to crates/cujson-sys/cuda/lines_chunks.h with a tier-1 test (crates/cujson-sys/tests/lines_chunks.rs) that fails on the old logic
- Second run (after 10e53f1 and fdee882): 13/14 checks pass; Lines mode with one chunk now passes the tape diff and all 51 per-line `to_value()` comparisons

## Missing

- Lines mode with 4096-byte chunks (51 single-line chunks) fails the GPU parse with `CUDA error 101: cudaErrorInvalidDevice: invalid device ordinal` — no call in parse_json_lines.cu, parse_standard_json.cu or utils.cu names a device ordinal (grep for cudaSetDevice, cudaGetDevice, cudaDeviceGetAttribute, cudaGetDeviceProperties finds none), so the failing call is not yet identified
