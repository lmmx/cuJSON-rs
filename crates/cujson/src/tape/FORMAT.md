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

`cuJSONResult::depth` (`cujson_types.h:30`) is read by the iterator as `jsonDepth`
(`query_iterator_standard_json.cpp:99`) but no write site for `parsed_tree.depth` was
found in `parse_standard_json.cu` or `parse_json_lines.cu` in the portions read for this
task (see Unresolved) — `Parser`'s internal `depth_init_MathAPI` +
`thrust::inclusive_scan` (`parse_standard_json.cu:1534-1543`) computes a *per-opener*
depth value used only to sort brackets for pairing, and is not obviously the same value
copied out to `cuJSONResult::depth`. The CPU reference builder computes `depth` as the
maximum bracket-nesting depth of the document (root array/object at depth 1, consistent
with `jsonDepth` starting semantics implied by `node_depth = 1` comments at
`query_iterator_standard_json.cpp:43,269`), and flags this as a hypothesis.

## 5. JSON Lines (`parse_json_lines`)

Each chunk (one JSON value per line) is parsed independently by the same
tokenizer/parser pipeline, with a running `lastStructuralIndex` (structural-entry count
so far) and `lastChunkIndex` (byte count so far) threaded into `stage2_tokenizer` so that
each chunk's structural offsets are expressed in the coordinate space of the whole
concatenated `input.data` (`parse_json_lines.cu:1190,1231-1232`) — i.e. structural offsets
are **global**, not per-chunk-relative.

Per chunk `i`, `resultSizes[i]` is that chunk's structural-entry count and
`resultSizesPrefix[i]` is the running total after chunk `i`
(`parse_json_lines.cu:1226-1228`). `mergeChunks` concatenates every chunk's structural
row into one buffer at `resultBuffer[1 + start_pos .. ]` where `start_pos =
resultSizesPrefix[i-1]` (`0` for `i == 0`), and every chunk's pair_pos row into
`resultBuffer[1 + start_pos + resultSizesPrefix[last] + 1 .. ]`
(`parse_json_lines.cu:1087-1095`). This reproduces the single-document layout — one
leading artificial `0`, the concatenated structural offsets, then (after the same
`resultSizesPrefix[last] + 1` gap used for the single-document `pair_pos` offset) the
concatenated pair_pos values — over the whole multi-chunk tape, with **no artificial
trailing close appended per chunk**: chunk boundaries are visible only through
`resultSizesPrefix`, not through extra tape entries. (`parse_standard_json.cu`'s
per-chunk `pair_pos` values, computed with `lastStructuralIndex` already added per
`parse_json_lines.cu:1203`, land directly in final tape-index space, so no
renumbering happens in `mergeChunks`.)

`totalResultSize = total_result_size + 2` and `fileSize = lastStructuralIndex + 2`
(`parse_json_lines.cu:1262-1263`) are equal here (`lastStructuralIndex` ends at
`total_result_size`), unlike the field-name difference implied in §2 — both count the
final tape length including the two artificial entries.

A `\n` byte landing at a structural offset (i.e. the newline used to separate two JSON
values inside one chunk, or possibly a raw `\n` chosen as a chunk's delimiter) is read
back as `','` by `getChar` (`query_iterator_standard_json.cpp:161-163`), effectively
making consecutive per-line documents look like elements of one array/stream to the
navigator. The CPU reference builder's `Mode::Lines` reproduces this: it records a
structural entry at each line-separating `\n` and the built tape's `getChar`-equivalent
returns `,` for it, matching the kernel path bit for bit in intent though the *value*
stored in `structural[i]` is the real byte offset of the `\n`, not a synthetic comma
byte — consistent with §1/§2 (the offset is stored, translation to `,` happens at read
time).

`Document::lines()` (this crate) does not attempt to reconstruct `cuJSONLinesInput`
chunking; it walks the merged, single flat tape and splits at top-level commas exactly
as `getChar` would present them, which is observably equivalent for chunks that contain
exactly one JSON value each (the documented use case — `cujsonlines.h`, not fully read
in this task).

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

1. **`cuJSONResult::depth` write site not located.** No assignment to `parsed_tree.depth`
   was found in `parse_standard_json.cu` or `parse_json_lines.cu`. It may be set in a
   file not read for this task, or it may simply be left as default-initialised garbage
   upstream. The CPU reference builder computes it as max bracket-nesting depth (root =
   depth 1) as a best-effort value; **this is a hypothesis about intended semantics, not
   a verified match to kernel output.**
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
5. **JSON-Lines global-offset claim (§5)** — that `lastChunkIndex`/`lastStructuralIndex`
   thread through `stage2_tokenizer` to produce globally-addressed structural offsets —
   is inferred from the accumulator variables' names and update sites
   (`parse_json_lines.cu:1231-1232`) and from `mergeChunks`' flat concatenation
   (`parse_json_lines.cu:1087-1095`); `stage2_tokenizer`'s own body (in a file not
   fully read for this task) was not inspected to confirm it actually adds
   `lastChunkIndex` to each byte offset it emits.
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
