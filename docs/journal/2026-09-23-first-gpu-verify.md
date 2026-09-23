# 2026-09-23: First GPU Runs of `cujson verify`

Host: NVIDIA GeForce RTX 3090 (sm_86), CUDA runtime 13020, driver 13020, `CUJSON_CUDA_ARCHS=86 cargo gpu-verify` (release build, 31 s).

## Current State

- Standard mode passes all three checks on the large-record fixture — GPU parse, `diff_tapes` against the CPU reference, and `to_value()` against `serde_json` (tier 3) — so FORMAT.md §1–§3 match the kernels' output for this input
- Error recovery passes in both modes: invalid UTF-8 returns `InvalidUtf8`, `{"a":[1,2}` returns `Unbalanced`, and a following valid parse passes the tape diff (tier 3)
- Free device memory reads 23,577,493,504 bytes before and after 700 mixed parses (300 valid, 300 invalid, 100 multi-chunk lines) in the second run, and 23,622,975,488 bytes before and after in the first — delta 0 both times, consistent with the lane A leak fixes (tier 3; heuristic, since the stream-ordered pool can hold memory)
- First run (commit 51a5f33): Lines mode with one chunk failed the tape diff at index 22439 (`structural[len-1]`: GPU 0, CPU 22439), and Lines-mode "valid parse after errors" failed at index 0 (GPU 2119, CPU 0) — `mergeChunks` never writes either artificial wrapper entry (parse_json_lines.cu:1093-1113); commit fdee882 makes the shim write them, and FORMAT.md §5 records the correction
- First run: Lines mode with 4096-byte chunks returned `Error::Internal` — the shim's chunker emitted a zero-size first chunk whenever the first line exceeded `chunk_bytes` (fixture lines average ~6 KB), which parse_json_lines rejects by returning an empty result; commit 10e53f1 moves the chunker to crates/cujson-sys/cuda/lines_chunks.h with a tier-1 test (crates/cujson-sys/tests/lines_chunks.rs) that fails on the old logic
- Second run (after 10e53f1 and fdee882): 13/14 checks pass; Lines mode with one chunk now passes the tape diff and all 51 per-line `to_value()` comparisons
- compute-sanitizer on the 4096-byte-chunk run reports "Grid Dimension X must be nonzero" on a kernel launch in `cujson_lines::stage3_parser`, then `cudaErrorInvalidValue` from `cudaLaunchKernel`, which the next Thrust call (`inclusive_scan`) picks up and throws — the "error 101" in `verify`'s output came from that Thrust exception, not from a device-ordinal call
- The small-records fixture separates records with blank lines (103 lines: 51 records, 52 empty); at 4096-byte chunking each record is its own chunk and each lone `\n` becomes a chunk with zero brackets, so `oc_cnt == 0` and `numBlock_open_close_32 == 0`
- `stage3_parser` (parse_json_lines.cu) and `Parser` (parse_standard_json.cu) return the structural row unchanged when `oc_cnt == 0`, freeing the same buffers as their success paths — the fix is compiled (tier 2) but not yet run on the GPU
- `verify`'s default sweep adds three bracket-free inputs: JSON Lines `[1]\n\n2\n"x"\n{"a":null}\n` at `chunk_bytes = 1`, and standard documents `42` and ` "s" `; the CPU reference yields tape `[0, 1]` for both scalars and navigates it to `42` / `"s"` (tier 1)

- The third run (after 29d6505 and a2f50c5) still failed the four bracket-free inputs with the same error, and its build printed "All library kernels up-to-date, skipping compilation" — `.watch(["cuda"])` hashes only `.h`/`.cuh`/`.hpp` files under a directory (cudaforge src/hash.rs `hash_paths`), so the guard in the `#include`d upstream `.cu` files was never compiled; commit 29aefa3 passes every file under `cuda/` to `.watch()` explicitly, and an edit to an included `.cu` or `.cpp` file now recompiles all three objects (tier 2, observed in this container)
- Fourth run (after 29aefa3, which recompiled 3 of 3 kernels): 25/25 checks pass — standard fixture, JSON Lines fixture at one chunk and at 4096-byte chunks, blank and scalar lines at one line per chunk, top-level `42` and ` "s" `, error recovery in both modes, and zero free-memory change over 700 parses (tier 3)

## Missing

- No compute-sanitizer memcheck run over the full `verify` sweep, so out-of-bounds device accesses that do not change results are unexamined
- No GPU other than sm_86 and no CUDA 12.x host has run `verify`; the PTX fallback path is untested
