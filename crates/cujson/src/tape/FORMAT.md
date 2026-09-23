# cuJSON tape format

Derived from `crates/cujson-sys/cuda/upstream/` at the imported commit
(`38d27b6e6c4eb74205cf59f4123b0983034405e2`, see `UPSTREAM.md`). Every rule below cites
the upstream file:line it was read from. Paths are relative to
`crates/cujson-sys/cuda/upstream/`.

A "tape" is the pair of `int32` arrays `cuJSONResult::structural` and
`cuJSONResult::pair_pos` (`cujson_types.h:28-29`), plus the token count `fileSize`
(`cujson_types.h:32`) and `depth` (`cujson_types.h:30`, max nesting depth — no kernel
site computing it was located in this task's reading; see Unresolved).

## 1. Structural character set

A byte is a candidate structural character iff it is one of `{ } [ ] : ,`
(`parse_standard_json.cu:447-571`, the `res_op` bitmap built by `bitMapCreatorSimd` —
this is the kernel actually launched, at `parse_standard_json.cu:1095`; the earlier,
never-launched `bitMapCreator` at `parse_standard_json.cu:330-400` is dead code and must
not be cited). `res_op`'s candidate set is `{ } [ ] : ,` only: the bracket class is built
at `parse_standard_json.cu:475-479`/`600-604` (`temp_open_close`/`temp2_open_close`) and
the colon/comma class at `:485-487`/`532-534` (`temp_colon_comma`/`temp2_colon_comma`);
a `\n`-including variant of that colon/comma class exists in the source but is commented
out (`parse_standard_json.cu:481-484`, `528-530`) and never executes. So the conclusion
holds unchanged: in Standard mode the structural set is exactly `{ } [ ] : ,`, and `\n`
is never itself structural. Bytes inside
a JSON string — including an escaped quote `\"` — are excluded: `findEscapedQuoteMerge_NEW`
computes `real_quote_GPU` as the set of *unescaped* quote bytes, handling runs of
backslashes with the standard odd/even-count rule (`parse_standard_json.cu:767-831`);
`inStringFinderBaseline` prefix-xors that bitmap to get an in-string mask
(`parse_standard_json.cu:851-859`); `findOutUsefulStringMerge` intersects the candidate
set with the complement of that mask: `inString_GPU[k] = ~in_string & all_structural`
(`parse_standard_json.cu:925-929`). So the final structural set is exactly the `{ } [ ] : ,`
bytes that lie outside any string body.

Whitespace (space, tab, CR, LF, or any other byte) is never itself structural in Standard
mode, full stop — including LF. `bitMapCreatorSimd`'s `open_close_GPU` output
(`parse_standard_json.cu:503,569`, misleadingly commented `// \n` at :503, a leftover
from the JSON-Lines variant of this same function name) is actually the **bracket-only**
bitmap (built from `temp_open_close`/`temp2_open_close`, i.e. `{ } [ ]` only) fed to
`depth_init_MathAPI` and bracket pairing (§3) — it carries no newline information in
Standard mode. There is no newline bitmap at all on the Standard-mode live path; the
JSON-Lines kernel of the same name (§5) is a different function, in a different
translation unit, that really does fold `\n` into its structural bitmap.

## 2. `structural` array layout

For a single document (`parse_standard_json`), let `result_size` be the number of real
structural bytes found. The host array `res_buff` (returned as `structural`) has
`result_size + 2` entries (`parse_standard_json.cu:1663,1676`):

- `structural[0] = 0` — artificial entry (`parse_standard_json.cu:1677`, re-asserted by
  the iterator constructor at `query_iterator_standard_json.cpp:92`)
- `structural[1 .. result_size]` — the real structural byte offsets, **1-based**:
  `res_buff[1..]` is a verbatim device→host copy of the Parser's structural-offset row
  (`parse_standard_json.cu:1666`); consumers read `input[structural[i] - 1]`
  (`query_iterator_standard_json.cpp:154`)
- `structural[result_size + 1] = totalResultSize - 1 = result_size + 1` — artificial
  entry, i.e. its own tape index (`parse_standard_json.cu:1678`, re-asserted at
  `query_iterator_standard_json.cpp:95`)

`fileSize = result_size + 2` (`parse_standard_json.cu:1676`) — **`fileSize` is a token
(structural-entry) count, not a byte count**; confirmed both by that assignment and by
an existing comment in the iterator itself: "`fileSize` counts structural tokens, not
bytes; `len` is the real JSON buffer length and is the correct bound for indexing
`inputJSON`" (`query_iterator_standard_json.cpp:155-156`).

`totalResultSize = result_size + 2` (`parse_standard_json.cu:1675`) — equal to
`fileSize` for the single-document path; kept as a separate field because the JSON-Lines
path (§5) gives them different values.

### Reading a tape entry as a character (`getChar`)

`getChar(idx)` (`query_iterator_standard_json.cpp:148-165`):
- `idx == 0` → `'['` (artificial open, regardless of what is stored at `structural[0]`)
- `idx == totalResultSize - 1` → `']'` (artificial close)
- otherwise: `pos = structural[idx] - 1`; bounds-checked against `len` (real buffer
  length), not `fileSize`; if `inputJSON[pos] == '\n'` returns `','` (JSON-Lines
  newline-as-comma, confirmed structurally even though the single-document path never
  puts a real `\n` at a structural offset); otherwise returns `inputJSON[pos]` verbatim.

So `structural[0]` and `structural[last]`'s *stored values* (`0` and `result_size+1`)
are never dereferenced into `inputJSON` — `getChar` special-cases those two indices
before computing `pos`. The CPU reference builder stores the same sentinel values for
byte-for-byte tape compatibility, but a navigator implementation may equally treat
index `0`/`last` as `'['`/`']'` without reading `structural[0]`/`structural[last]` at all.

## 3. `pair_pos` array

`pair_pos` is written only for **opening-bracket** structural indices, by
`validate_expand_MathAPI_new2` (`parse_standard_json.cu:1435-1517`), which is launched
over the (already depth-sorted) list of `{ } [ ]` occurrences only — `,`/`:` never reach
this kernel (`parse_standard_json.cu:1548-1549` builds `zipped_begin` from `oc_idx` /
`open_close_GPU`, the bracket-only stream produced earlier in `Parser`). For each
open/close bracket pair at local (0-based, per-chunk) indices `k`/`k+1` in that stream:

```
endIdx[index_arr[k]] = index_arr[k+1] + lastStructuralIndex + 1;
```
(`parse_standard_json.cu:1468,1487-1488,1510-1511`) — i.e. `pair_pos[<opener's final
tape index>] = <closer's final tape index>`. `index_arr[k]` is already expressed in
final (1-based-into-structural, chunk-offset-adjusted) tape-index space, matching how
`res_buff[1..]` was populated.

**`pair_pos` for `,`, `:`, and closing brackets is never written by any kernel** — those
tape slots hold whatever the pinned host allocation happened to contain
(`cudaMallocHost` at `parse_standard_json.cu:1663` performs no `cudaMemset`/zero-fill).
This is the answer to the brief's open question: pair_pos is only meaningful for
openers. Consumers only ever call `jumpOpeningForward(idx)` (which reads `pair_pos[idx]`,
`query_iterator_standard_json.cpp:167-178`) from an index that `getChar` has confirmed is
`[` or `{` (see every call site in `getKey`/`getValue`/`findKey`,
`query_iterator_standard_json.cpp:494-604`), so the undefined entries are never read
through the intended API — but they are undefined, not zero, and the CPU reference
builder does not need to (and does not) reproduce whatever garbage the GPU allocator
happened to leave there. See Unresolved.

Two boundary entries are explicitly overwritten by the iterator constructor, not the
kernel:
- `pair_pos[0] = totalResultSize - 1` (`query_iterator_standard_json.cpp:97`) — the
  artificial open pairs with the artificial close.
- `pair_pos[totalResultSize - 1]` (the artificial close's own pair entry) is **not**
  set by the constructor and not written by the kernel — undefined. The CPU reference
  builder sets it to `0` (pairing back to the artificial open) since that is the only
  value consistent with `pair_pos[0]`'s assignment and with treating the whole tape as
  one bracket pair; this is a builder policy choice, not an observed kernel behaviour.

## 4. `depth`

**Finding: upstream never writes `cuJSONResult::depth`.** `grep -n '\.depth\|depth ='` over
`parse_standard_json.cu` and `parse_json_lines.cu` finds exactly two matches, and both are
an unrelated *local* variable that happens to share the name, aliasing the bracket-pairing
scratch buffer: `uint32_t* depth = oc_1;` (`parse_standard_json.cu:1538`,
`parse_json_lines.cu:1033`). Neither file ever assigns to `parsed_tree.depth` or any other
`cuJSONResult`-typed field named `depth`. `depth_init_MathAPI` +
`thrust::inclusive_scan`/`transform_if` (`parse_standard_json.cu:1534-1543`,
`parse_json_lines.cu:1030-1037`) compute a *per-opener* nesting depth used only to sort
brackets before pairing (`thrust::stable_sort_by_key` on that same buffer,
`parse_standard_json.cu:1548`-ish/`parse_json_lines.cu:1042`) — it is freed
(`cudaFreeAsync(depth, 0)`) rather than copied out anywhere.

Both construction sites — `cuJSONResult parsed_tree;` (`parse_standard_json.cu:1589`,
`parse_json_lines.cu:1103`) — are plain default-initialization of an aggregate struct with
no user-declared constructor and no default member initializers (`cujson_types.h:22-32`),
so `depth` (and every other scalar field) reads as **indeterminate stack garbage** on the
success path, not zero. The only place this struct is zero-initialized is the early-error
`return cuJSONResult{};` value-initialization on the validation-failure paths (e.g.
`parse_json_lines.cu:1114,1119,1124,1129`), which never reaches a real parse.

`cuJSONResult::depth` (`cujson_types.h:30`) is nonetheless *read* by the iterator as
`jsonDepth` (`query_iterator_standard_json.cpp:99`) — so the iterator trusts a field the
kernel never sets, on the assumption some other unread code path sets it, or that it was
never exercised. The Rust builder still computes its own `depth` (max bracket-nesting
depth, root at depth 1, consistent with `node_depth = 1` comments at
`query_iterator_standard_json.cpp:43,269`) because the navigator needs *some* depth value
internally, but this field has **no GPU counterpart to diff against** — task 10's
differential test must exclude `Tape::depth` from its comparison entirely, not just mask
it, since there is no defined kernel value to compare it to (see `Tape::depth`'s doc
comment in `storage.rs`).

## 5. JSON Lines (`parse_json_lines`)

`parse_json_lines` is a **different pipeline** from Standard mode, in `parse_json_lines.cu`
— it does not reuse the Standard kernels. The host loop (`parse_json_lines`,
`parse_json_lines.cu:1101-1273`) splits the input into `input.chunkCount` chunks (each
containing one or more whole lines — see the chunk-boundary note below), and for each
chunk `i` launches, in order:

- `checkAscii` (`:272`) / `checkUTF8` (`:287`) — UTF-8 validation, via `stage1_UTF8Validator`.
- `bitMapCreatorSimd` (`:849`, function body `:423-536`) — **this is a distinct function
  from the Standard-mode `bitMapCreatorSimd` in `parse_standard_json.cu`** (same name,
  different translation unit, different behaviour): its structural class
  (`temp_colon_comma_newline`/`temp2_colon_comma_newline`, `:453-456`/`:495-498`)
  *includes* `0x0A` (`\n`), uncommented and live — unlike the Standard-mode function,
  where the equivalent line is commented out (§1). So in Lines mode, every unescaped `\n`
  outside a string is structural, exactly like `{ } [ ] : ,`.
- `fusedStep2_3` (`:857`) / `buildStringMask` (`:870`) — same quote/string-mask logic as
  Standard mode, applied to this chunk.
- `fusedStep3_4` (`:876`, body `:695-721`) — masks the structural bitmap (which now
  includes `\n`) by the string mask: `str_mask[k] = ~curr_str_mask & all_structural`
  (`:709`) — so a `\n` *inside* a string is excluded exactly like a bracket or comma
  inside a string would be (§1's rule applies uniformly, `\n` included).
- `extractStructuralIdx` (`:918`, body `:740-804`) — turns the masked bitmap into offsets.
  Line `:792`, `out_string_8_index_GPU[adjusted_index] = k + j + 1 + lastChunkIndex;`,
  **confirms** the global-offset claim below by direct inspection (this resolves what was
  previously listed as Unresolved #5: `lastChunkIndex` is added to every emitted offset,
  not merely inferred from variable names).
- `map_open_close` (`:1029`, body `:946-958`) / `validate_expand` (`:1056`, body
  `:961-1015`) — bracket-only pairing for this chunk. Note the name: it is
  `validate_expand`, not `validate_expand_MathAPI_new2` (that name belongs to a different
  file/mode and must not be cited here). `endIdx[index_arr[k]] = index_arr[k+1] +
  lastStructuralIndex + 1;` (`:983`, `:995-996`, `:1007-1008`) is the same
  opener-index-to-closer-index write pattern as Standard mode's `validate_expand`
  counterpart (§3), with `lastStructuralIndex` folding in this chunk's running offset.

`stage2_tokenizer` (`:808-943`) and `stage3_parser` (`:1016-1074`) are both fully defined
in this file — an earlier draft of this document called them "not fully read"; that was
wrong. `stage2_tokenizer` runs the five kernels above in order and returns structural
offsets already in global (whole-`input.data`) coordinate space, via the `lastChunkIndex`
add at `:792`. `stage3_parser` runs `map_open_close`/`validate_expand` and returns a
chunk-local buffer laid out as `[structural row][pair_pos row]`, both already expressed
in *global* tape-index space because `lastStructuralIndex` was folded into `validate_expand`
at `:1056` (`stage3_parser`'s own `lastStructuralIndex` parameter, `:1016`).

Per chunk `i`, `resultSizes[i]` is that chunk's structural-entry count and
`resultSizesPrefix[i]` is the running total after chunk `i`
(`parse_json_lines.cu:1226-1228`). `mergeChunks` (`:1076-1096`) concatenates every chunk's
structural row into one buffer at `resultBuffer[1 + start_pos ..]` where `start_pos =
resultSizesPrefix[i-1]` (`0` for `i == 0`, `:1088-1089,1092`), and every chunk's pair_pos
row into `resultBuffer[1 + start_pos + resultSizesPrefix[last] + 1 ..]` (`:1094`). This
reproduces the single-document layout of real entries — the concatenated structural
offsets, then (after the same `resultSizesPrefix[last] + 1` gap used for the
single-document `pair_pos` offset) the concatenated pair_pos values — over the whole
multi-chunk tape, with **no artificial trailing close appended per chunk**: chunk
boundaries are visible only through `resultSizesPrefix`, not through extra tape entries.
Because both the structural offsets (`lastChunkIndex`, confirmed above) and the pair_pos
values (`lastStructuralIndex`, confirmed above) are already in final tape-index/byte-space
before `mergeChunks` runs, `mergeChunks` does pure concatenation — no renumbering.

**`mergeChunks` never writes the two artificial wrapper entries.** `resultBuffer[0]`
(`structural[0]`) and `resultBuffer[N+1]` (`structural[len-1]`, the same int32 as
`pair_pos[0]`) are left as whatever the non-zeroing `cudaMallocHost` returned
(`parse_json_lines.cu:1093-1113`); upstream's iterator constructor writes them itself
(`query_iterator_standard_json.cpp:92,95,97`). An earlier revision of this document said
`mergeChunks` wrote a leading artificial `0` — the first GPU run (RTX 3090, 2026-09-23)
refuted that, reading `structural[len-1] = 0` on a fresh allocation and `structural[0] =
2119` on a reused one. The C ABI shim therefore writes `structural[0] = 0` and
`structural[len-1] = len-1` after `parse_json_lines` returns (`capi_lines.cu`), matching
Standard mode (§2), so the tape the Rust crate receives is fully defined in both modes.

`totalResultSize = total_result_size + 2` and `fileSize = lastStructuralIndex + 2`
(`parse_json_lines.cu:1262-1263`) are equal here (`lastStructuralIndex` ends at
`total_result_size`), unlike the field-name difference implied in §2 — both count the
final tape length including the two artificial entries.

A `\n` byte landing at a structural offset is read back as `','` by `getChar`
(`query_iterator_standard_json.cpp:161-163`), effectively making consecutive per-line
documents look like elements of one array/stream to the navigator. The CPU reference
builder's `Mode::Lines` reproduces this with a single whole-input scan (`builder.rs`'s
`scan_structural(.., mark_newline: true)`) rather than a per-line split: since brackets
never span a `\n` in valid per-line JSON, a stack-based scan that also marks every
unescaped-outside-a-string `\n` structural produces the identical structural/pair_pos
content as the kernel's per-chunk-then-merge process, without needing to reproduce
chunking at all (see the newline-case rules below).

### Newline semantics (four cases)

All four are determinable from the kernel/loader source read for this task; none require
guessing.

1. **Trailing `\n` at EOF.** `bitMapCreatorSimd`'s structural class has no end-of-input
   special case for `\n` beyond the generic last-word tail handling shared by every other
   structural byte (`:461-472`). A trailing `\n` gets its own structural entry, read back
   as `,` by `getChar`, immediately before the tape's artificial closing entry. The CPU
   builder matches this exactly (the whole-input scan doesn't special-case a trailing
   `\n` either). `Document::lines()` (navigator layer, not tape format) filters out the
   resulting empty scalar span rather than yielding a phantom trailing value — see its
   doc comment in `document.rs`.
2. **Blank lines / consecutive `\n\n`.** Same reasoning: each `\n` is independently
   structural, unconditionally — nothing in `bitMapCreatorSimd`, `fusedStep3_4`, or
   `extractStructuralIdx` merges or dedupes adjacent structural bits. Two consecutive
   `\n` bytes give two consecutive structural entries, both reading back as `,`, with
   nothing between them. The CPU builder matches this exactly; `Document::lines()` again
   filters the resulting empty span.
3. **CRLF (`\r` immediately before `\n`).** `\r` is `0x0D`; no comparison against `0x0D`
   appears anywhere in `bitMapCreatorSimd` (Standard or Lines variant) or any other
   tokenizer kernel in either `.cu` file (confirmed by grep — the only `\r`-related text
   in either upstream file is a commented-out line, `parse_standard_json.cu:405`). `\r`
   is therefore never structural and never specially skipped; it is ordinary
   non-structural whitespace, identical in treatment to a space or tab byte. The CPU
   builder already treats it this way (falls through `scan_structural`'s `_ => {}` arm).
4. **A chunk boundary falling near/at a newline.** This is a host-loader property, not a
   kernel one: `loadJSONLines_chunkCount`/`_chunkSizeBytes`/`_chunkSizeMegaBytes`
   (`load_file.cu`) build a `line_offsets` table by scanning for `\n` bytes first, then
   only ever cut a chunk boundary at one of those offsets (`load_file.cu`'s
   `start_offset`/`end_offset = line_offsets[...]` in the chunk-count loader, and the
   `if ((line_end - current_chunk_start) > chunkSizeBytes)` cut-on-line-boundary check in
   the byte/megabyte loaders) — never mid-line, never splitting a `\n` from the byte
   before it. A `\n` is therefore always fully contained in exactly one chunk. Combined
   with `lastChunkIndex`/`lastStructuralIndex` correctly threading chunk-relative results
   into global tape coordinates (confirmed above), a multi-chunk parse of some input is
   provably equivalent, structural-entry for structural-entry, to a single-chunk parse of
   the same input — chunking is a pure parallelism detail with no visible effect on tape
   content. The CPU reference builder never chunks at all (one call, one buffer), which is
   exactly the "chunk count = 1" case of this equivalence, so no chunk-boundary-specific
   code is needed in `builder.rs`; `lines_chunk_boundary_is_a_noop_for_cpu_builder` in
   `crates/cujson/tests/tape_tests.rs` records this reasoning as a test.

`Document::lines()` (this crate) does not attempt to reconstruct `cuJSONLinesInput`
chunking (chunking has no effect on tape content per case 4 above); it walks the merged,
single flat tape and splits at top-level commas exactly as `getChar` would present them.

## 6. Empty containers and top-level scalars

An empty `{}` or `[]` is two structural entries (the two brackets) — nothing about the
tokenizer or parser special-cases zero-content containers; `res_op`'s bitmap includes
open and close of an empty pair identically to a non-empty one
(`parse_standard_json.cu:393-400`). The CPU reference builder emits an opener
immediately followed by its matching closer, with `pair_pos[opener] = closer`, no
different from the non-empty case.

**Top-level scalar documents** (e.g. input `42` or `"x"` with no enclosing `{}`/`[]`):
nothing in `res_op` marks a bare scalar as structural (§1's set is only brackets/`:`/`,`),
so `result_size` would be `0` for such an input, giving a 2-entry tape
(`structural = [0, 1]`, i.e. only the two artificial brackets, `pair_pos[0] = 1`). No
code path in `parse_standard_json.cu` rejects this up front, but the two artificial
entries mean the navigator sees an empty array/object shape, not a scalar — **the
kernel's tape format cannot distinguish a top-level scalar from an empty container**,
since neither produces any real structural entries and the artificial wrapper always
reads as `[`/`]`. The CPU reference builder treats a top-level scalar as a special case
outside the wrapper (see Unresolved) rather than silently producing an indistinguishable
empty-array tape, because the two are semantically different documents.

## Unresolved

These are open because the upstream source read for this task does not settle them, or
settles them inconsistently between kernel and iterator. Do not guess past what is
written here.

1. ~~`cuJSONResult::depth` write site not located.~~ **Resolved as a finding, see §4**:
   upstream genuinely never writes it (confirmed by grep), and the struct is default- not
   value-initialized on the success path, so the field is indeterminate garbage upstream.
   The Rust builder's own `depth` has no GPU value to match and must be excluded from
   task 10's differential test.
2. **`pair_pos` for non-opener entries is undefined**, not just "not part of the public
   API" — `cudaMallocHost` does not zero-fill, and no kernel writes those slots (§3). A
   differential test against real GPU output (task 10) cannot assert equality on those
   slots; it should mask them out.
3. **`pair_pos[totalResultSize-1]`** (the artificial closer's own entry) is not
   initialised by the kernel or the iterator constructor. The CPU builder's choice of
   `0` here is a policy decision (§3), not an observed fact — task 10's differential
   test must not assert on it.
4. **Whitespace skipped by the iterator is only `' '`** (`jumpSpacesForward`/
   `jumpSpacesBackward`/`jumpValueForward`/`jumpValueBackward`,
   `query_iterator_standard_json.cpp:180-216` all test only `inputJSON[pos] == ' '`).
   Tab/CR/LF inside otherwise-scalar whitespace are **not** skipped by these helpers —
   a scalar preceded/followed by `\t` rather than `' '` would have that tab byte
   included in `getValue`'s returned span by the reference C++ iterator. This is very
   likely an upstream bug (JSON permits tab/CR/LF as insignificant whitespace) rather
   than intended behaviour, but this task's brief says "follow the kernel behaviour"
   only where kernel and iterator disagree — here they agree (the kernel doesn't treat
   tab/CR as structural either, so it never produces a structural offset landing on
   one; the risk is only in cross-token whitespace runs). The CPU reference builder
   trims **all** JSON insignificant whitespace (space, tab, CR, LF) from scalar `raw()`
   spans, which is a deliberate divergence from the literal C++ `jumpSpaces*` behaviour,
   noted here rather than silently matched, because reproducing a plausible bug was
   judged out of scope for a from-scratch Rust builder. Fixture/property tests in this
   task only use space-separated/compact/pretty (`serde_json`-formatted, LF+space)
   JSON, so this divergence is not exercised by tier-1 tests.
5. ~~JSON-Lines global-offset claim (§5).~~ **Resolved**: `stage2_tokenizer`'s body was
   read in full; `extractStructuralIdx` (`parse_json_lines.cu:792`) directly adds
   `lastChunkIndex` to every emitted structural offset, and `validate_expand`
   (`:983,995-996,1007-1008`) directly adds `lastStructuralIndex` to every pair_pos
   value — both confirmed by inspection, not inferred from names.
6. **Top-level scalar documents (§6)**: whether `parse_standard_json` is meant to
   support them at all, versus always expecting an object/array root, is not stated by
   any comment or check in the ~1550-1690 assembly region read for this task. The CPU
   reference builder supports them (as a value directly, without needing bracket
   wrapping) as the more useful behaviour for a library, and records this as a
   **divergence from a hypothesis about kernel behaviour**, not a divergence from a
   confirmed fact.
7. **Whether `pair_pos`'s two host-side writes for a bracket pair
   (`endIdx[index_arr[k]] = …` at `parse_standard_json.cu:1468` and `:1487-1488`,
   `:1510-1511`) can race or duplicate-write** across thread blocks for adjacent
   4-wide batches was not analysed; assumed correct since it's unmodified upstream
   kernel code.

Byte-for-byte equality between a CPU-reference tape and a real GPU-produced tape for the
same input is **UNVERIFIED** — this document and the CPU builder are derived entirely
from static reading of the CUDA/C++ source with no GPU available in this container. It
remains a hypothesis until task 10's differential runbook confirms it on a GPU box
(tier 3).
