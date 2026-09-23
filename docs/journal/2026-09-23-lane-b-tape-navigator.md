# 2026-09-23: Tape Format Spec, CPU Reference Builder, and Navigator (task 05)

## Current State

- `crates/cujson/src/tape/FORMAT.md` specifies the tape format derived from
  `crates/cujson-sys/cuda/upstream/{parse_standard_json.cu,parse_json_lines.cu,query/query_iterator_standard_json.cpp}`,
  citing file:line for every rule, with an "Unresolved" section (7 items) for
  what the source leaves ambiguous or unwritten
- `crates/cujson/src/tape/storage.rs` defines `TapeStorage` (an enum with one
  `Owned(Box<[i32]>)` variant behind `Deref<Target = [i32]>`) and `Tape {
  structural, pair_pos, depth }`; the enum is the documented extension point
  for task 06 to add a pinned-allocation variant freed via `cujson_tape_free`
  on `Drop`, without changing navigator code
- `crates/cujson/src/tape/builder.rs` (behind the `cpu-reference` feature)
  implements `build_tape_cpu(input: &[u8], mode: Mode) -> Result<Tape, Error>`
  for `Mode::Standard` and `Mode::Lines`, producing exactly the tape defined
  in FORMAT.md: a sequential scan tracking string/escape state to find
  structural bytes (`scan_structural`), bracket-pair matching via a stack,
  and (for `Mode::Lines`) per-line scanning spliced with a structural entry
  at each separating `\n`
- `crates/cujson/src/tape/document.rs` implements `Document<'a> { input:
  Cow<'a, [u8]>, tape: Tape }` and `Node<'d>` with `kind`, `get`, `index`,
  `len`, `iter_array`, `iter_object`, `raw`, `as_str` (unescaping, including
  `\uXXXX` and surrogate pairs), `as_f64`, `as_i64`, `as_bool`, `is_null`;
  `Document::pointer` implements RFC 6901 (`~1`→`/`, `~0`→`~`, in that
  order); `Document::lines()` iterates top-level values for JSON Lines
  documents; `Node::to_value()` (behind the `serde` feature) converts to
  `serde_json::Value`, parsing numbers with `str::parse::<f64>`/`i64`/`u64`
  rather than `serde_json::from_str` (see Divergence)
- Child/pair iteration (`children_iter`, `object_pairs`,
  `crates/cujson/src/tape/document.rs`) walks the tape by alternating
  `read_value` (mirroring the upstream `getValue`'s dispatch on
  `currentNodeChar` being `,`/`:`/`[`, `query_iterator_standard_json.cpp:562-604`)
  with the delimiter that follows each value; container emptiness is decided
  by checking whether the trimmed byte span between the container's open and
  close tape indices is empty, not by tape-index adjacency (adjacency alone
  is ambiguous — `[null]`'s `[`/`]` tape indices are adjacent too, since a
  bare scalar gets no tape entry of its own)
- 11 tier-1 tests pass under `cargo test -p cujson --features
  cpu-reference,serde`: 6 unit tests over hand-worked tapes
  (`crates/cujson/src/tape/document.rs` `mod tests`), 3 `proptest` property
  tests (compact round-trip, pretty round-trip, `pointer()` vs
  `serde_json::Value::pointer()`) and 2 fixture tests over
  `tests/fixtures/*.json` (`crates/cujson/tests/tape_tests.rs`) — the large
  fixture as `Mode::Standard`, the small-records fixture as `Mode::Lines`
  via `Document::lines()`
- `crates/cujson/tests/tape_tests.rs` is gated with `#![cfg(all(feature =
  "cpu-reference", feature = "serde"))]` so `cargo test --workspace` (no
  features) still compiles the binary, with zero tests in it
- All five commands in the brief's Verification section pass: `cargo fmt
  --check`, both `cargo clippy --workspace --all-targets` invocations (with
  and without `cujson/cpu-reference,cujson/serde`) with `-D warnings`,
  `cargo test -p cujson --features cpu-reference,serde`, `cargo test
  --workspace` — all tier 1, no GPU in this container
- `proptest = "1"` added to the root `Cargo.toml` workspace dependencies;
  `crates/cujson/Cargo.toml` gained a `[dev-dependencies]` section with
  `proptest` and `serde_json` (both `workspace = true`)

## Missing

- No FFI-backed `TapeStorage` variant (explicitly out of scope — task 06's
  job; the doc comment on `TapeStorage` names the extension point)
- No full JSON grammar validation in `build_tape_cpu` beyond bracket
  balancing, string termination, and UTF-8 validity — e.g. `[1,,2]` or a
  bare `,` would not be rejected by the structural scan itself; property
  tests only exercise `serde_json`-serialised (always well-formed) input, so
  this gap is untested, not merely unverified

## Divergence

- The brief's "facts already observed" list states `pair_pos[i]` for an
  opening bracket gives the tape index of its closing bracket — confirmed,
  but with a correction: the brief does not mention that `pair_pos` is
  **only ever written by any kernel for opening-bracket indices**
  (`parse_standard_json.cu:1468,1487-1488,1510-1511`, all writes go through
  `validate_expand_MathAPI_new2`, which is launched only over the
  open/close-bracket stream). Entries for `,`, `:`, and the artificial
  closer are never written by device code and `cudaMallocHost`
  (`parse_standard_json.cu:1663`) does not zero-fill, so those slots are
  undefined on the GPU path — not merely "not part of the documented API".
  `build_tape_cpu` fills them with `-1` as an explicit sentinel rather than
  reproducing GPU-allocator garbage (`crates/cujson/src/tape/builder.rs`,
  see FORMAT.md §3 and Unresolved #2-#3)
- `Node::to_value()`'s number parsing does not use `serde_json::from_str`
  directly: doing so was found (during this task, via
  `crates/cujson/tests/tape_tests.rs`'s now-removed `debug_number_precision`
  scratch test) to be off by one ULP from `f64::from_str` for at least one
  generated value (`"980300.2658449085".parse::<f64>()` gives
  `980300.2658449085`; `serde_json::from_str::<Value>` on the same text
  gives `980300.2658449084`) — a `serde_json` 1.0.151 float-lexer quirk, not
  a tape/navigator bug. `to_value()` instead parses integers with
  `str::parse::<i64>`/`u64` and floats with `str::parse::<f64>` +
  `serde_json::Number::from_f64`

CPU-vs-GPU tape byte-for-byte equality is **UNVERIFIED** — everything above
was derived from static reading of the CUDA/C++ source with no GPU in this
container. It is a hypothesis until task 10's differential runbook confirms
it on a GPU box (tier 3); FORMAT.md's Unresolved section lists the specific
points (undefined `pair_pos` slots, unlocated `depth` write site, unverified
`stage2_tokenizer` global-offset behaviour for JSON Lines) that a
differential test cannot assert on even then.
