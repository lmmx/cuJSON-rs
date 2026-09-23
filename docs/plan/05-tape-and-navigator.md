# 05 — Tape format spec, CPU reference builder, and navigator

Depends on: 01. Lane B (parallel with lane A). Verification tier: 1 for everything here, except the claim that the CPU reference matches the GPU tape byte-for-byte, which is tier 3 (task 10).

This lane lets most of the library logic be built and tested without a GPU. The GPU produces a "tape" (two `int32` arrays); everything after that is ordinary host code.

Work only in `crates/cujson/src/tape/` (and `crates/cujson/tests/` for its tests). Don't touch `cujson-sys`. Reading `crates/cujson-sys/cuda/upstream/` is expected.

## Part 1: specify the tape (`crates/cujson/src/tape/FORMAT.md`)

Derive the format from the upstream sources, citing file:line for each rule:
- the kernels that produce `structural` / `pair_pos` (`parse_standard_json.cu`: `Tokenize`, `Parser`, `validate_expand_MathAPI_new2`, and the result assembly ≈1655-1690)
- the consumer `query/query_iterator_standard_json.cpp`, especially `getChar`, `jumpOpeningForward`, `gotoKey`, `gotoArrayIndex`, `getValue`, `getKey`

Facts already observed (verify them):
- `structural[0]` and `structural[len-1]` hold artificial entries read back as `[` and `]`, which wrap the document
- other `structural[i]` values are **1-based** byte offsets into the input (`getChar` reads `input[structural[i] - 1]`)
- the structural set includes `: ,` as well as brackets (the iterator navigates to colons and commas)
- in JSON Lines input a `\n` byte at a structural position reads as `,`
- `pair_pos[i]` for an opening bracket gives the tape index of its closing bracket
- `fileSize` holds a token count, not a byte count

Open questions to answer in FORMAT.md: what `pair_pos` holds for non-opener entries; which whitespace bytes are skipped (the iterator only skips `' '`); how JSON Lines chunk results get concatenated (`resultSizes`, `resultSizesPrefix`); whether a top-level scalar document is supported. When the source doesn't settle a question, say so explicitly rather than guessing.

## Part 2: CPU reference builder (`cpu-reference` feature)

`pub fn build_tape_cpu(input: &[u8], mode: Mode) -> Result<Tape, Error>` produces exactly the tape defined in FORMAT.md: same arrays, same artificial entries, same 1-based offsets. Straightforward sequential code; speed doesn't matter. It serves as (a) the fixture generator for navigator tests and (b) the oracle in task 10's differential test.

## Part 3: navigator

Types owned by this task:
- `Tape { structural: Box<[i32]> | borrowed pinned slice, pair_pos, depth }`. Abstract storage behind a small trait or enum so task 06 can back it with the pinned C allocation and free it with `cujson_tape_free`
- `Document<'a> { input: Cow<'a, [u8]>, tape: Tape }`
- `Node<'d>` with: `kind() -> Kind {Object, Array, String, Number, Bool, Null}`, `get(&str) -> Option<Node>`, `index(usize) -> Option<Node>`, `len()` for containers, `iter_array()`, `iter_object() -> (key: Cow<str>, Node)`, `raw() -> &[u8]` (the value's exact input bytes, whitespace trimmed), `as_str() -> Result<Cow<str>>` (unescapes), `as_f64`, `as_i64`, `as_bool`, `is_null`
- `Document::pointer(&str) -> Option<Node>` implementing RFC 6901 (`~0`/`~1` escapes)
- `serde` feature: `Node::to_value() -> serde_json::Value`

Scalars get located via the tape (they sit between two structural entries), then parsed from `raw()`. For strings, compare keys after unescaping so keys containing escapes still match.

JSON Lines documents: `Document::lines()` iterates the per-line roots.

## Tests (tier 1)

- Unit tests over hand-written tapes for small documents
- Property test (`proptest`): generate a random `serde_json::Value`, serialise it (compact and pretty), build the tape with `build_tape_cpu`, and check `root().to_value()` equals the original. Also check `pointer()` against `serde_json::Value::pointer` for random paths
- Run over both `tests/fixtures/*.json` files comparing against `serde_json`

## Acceptance

- `cargo test -p cujson --features cpu-reference,serde` passes; clippy is clean
- FORMAT.md cites a source line for every rule and lists the unresolved questions
- Journal entry states that CPU/GPU tape equality is unverified until task 10
