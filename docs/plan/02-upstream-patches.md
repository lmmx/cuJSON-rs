# 02 — Make upstream cuJSON library-safe

Depends on: 01. Lane A. Verification tier: 1 for Rust, **none for C++ here** (no nvcc). Tier 2 once task 09's CI exists. Keep every patch small and mechanical so it can be reviewed by reading it.

Edit only `crates/cujson-sys/cuda/upstream/`. One commit per numbered item.

## Patches

1. **Remove `<x86intrin.h>`** from `utils.cu`, `query/query_iterator.cpp` and `query/query_iterator_standard_json.cpp`. No intrinsics from that header are used (grep for `_mm`, `_pdep`, `_pext`, `__rdtsc`, `_popcnt`, `_tzcnt`, `_lzcnt` found none). Re-grep to confirm before removing.

2. **Replace process exits with error propagation.** Every `exit(` call in `parse_standard_json.cu` (1572, 1629), `parse_json_lines.cu` (1064, 1181) and the two query iterators (607, 557) must stop killing the host process. Approach: define `struct cujson_error { int code; }` and error codes (`CUJSON_ERR_UTF8 = 1`, `CUJSON_ERR_UNBALANCED = 2`, …) in a new header `cujson_error.h`, and replace each `exit(0)` with `throw cujson_error{CODE}`. Task 03's shim catches these exceptions at the C ABI boundary. **Before each throw, free the device and pinned allocations that are live at that point**, or the process leaks GPU memory on every invalid document. List the freed allocations per site in the commit message. Grep the whole upstream tree for `exit(`, `abort(` and `std::terminate` so no site is missed. Leave the query iterators' `exit` in place if task 05 makes them unused, but record that choice.

3. **Remove stdout/stderr output** from library paths: `printf("Incomplete ASCII!\n")` (parse_standard_json.cu:317), `printf("error found!")` (1571), the `std::cerr` lines in `load_file.cu`, plus any non-commented `printf`/`cout`/`cerr` in `parse_json_lines.cu` and `utils.cu`. Turn the ones that signal errors into item-2 throws.

4. **Input ownership: callers keep their buffers.**
   - `parse_standard_json` ends with `cudaFree(input.data)` (≈line 1689), but `input.data` points to host memory, so this call is invalid and leaves a pending CUDA error. Remove it.
   - `parse_json_lines` ends with `cudaFreeHost(input.data)` (≈line 1270), which frees the caller's buffer. Remove it.
   - After this patch both parse functions only read their input.

5. **Make both parsers linkable into one library.** `parse_standard_json.cu` and `parse_json_lines.cu` define the same non-static names (`vectorizedClassification`, `checkAscii`, `checkUTF8`, `bitMapCreator`, `bitMapCreatorSimd`, `continuationBytes`, `count_set_bits`, `findOutUsefulString`, and possibly others; generate the full list with a symbol diff). Both headers also use the same include guard, `STANDARD_PARSE_H`. Fix:
   - wrap each file's contents in its own namespace (`cujson_std` and `cujson_lines`), with system and thrust/cub includes staying outside the namespace
   - give `parse_json_lines.h` its own include guard
   - `utils.cu` is shared: either put it in a namespace included by both or mark its free functions `inline`/`static`, whichever produces the smaller diff

6. **Fix the misleading `fileSize` field.** `cuJSONResult::fileSize` is set to `result_size + 2`, a token count (parse_standard_json.cu ≈1683). Don't rename the field, but add a comment at the struct definition saying what it holds. Task 05 depends on this being documented.

## Out of scope

Performance changes, the `int` size limits (enforced in Rust instead), and CUDA error checking after individual API calls (task 03 checks once at the boundary).

## Acceptance

- `grep -rnE '\bexit\(|printf|std::cout|std::cerr|x86intrin' crates/cujson-sys/cuda/upstream` shows only commented lines or sites you justify in the journal entry
- The journal entry lists each patch and its commit, and states that none of it has been compiled yet (tier 0) unless nvcc was available
