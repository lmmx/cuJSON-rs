# cujson-bench

Times cuJSON against `simd-json` (the Rust port of simdjson used by `genson-rs`/`polars-genson`) on
the `claims` column of the Wikidata parquet dump, the workload the wikidata pipeline's
`normalise_claims_direct` runs. Not published.

The input, `data/chunk_0-00283-of-00546.parquet` (92 MB, 1818 rows, 233 MB of claims JSON), is
committed; `fetch.sh` re-downloads it from the pinned HuggingFace revision and checks its SHA-256.

```sh
# correctness gate: per-row checksums of simd-json vs cuJSON's CPU tape (and the GPU tape if built with cuda)
cargo run --release -p cujson-bench -- verify

# timing (cujson-* engines need the cuda feature and a GPU)
CUJSON_CUDA_ARCHS=86 cargo run --release -p cujson-bench --features cuda -- run
cargo run --release -p cujson-bench --features cuda -- run --engines cujson-visit-par --json
# the same walks over the sequential CPU-built tape, no GPU needed
cargo run --release -p cujson-bench -- run --tape cpu --levels walk
```

When `simd-par` and `cujson-pipe` both run at the `walk` level, the last line is the overall figure (the numbers here are an example):

```
overall (walk, whole file): simd-par 0.101s, cujson-pipe 0.045s: cujson-pipe is 2.24x faster
```

## What is measured

Each engine gets identical bytes: the column's rows joined as JSON Lines, in batches of at most
`--batch-mb` MiB (cuJSON rejects inputs of 2 GiB or more), built outside the timed region.
Two levels, per engine:

- `parse`: parse every row and discard the result
- `walk`: parse, then visit every node: kinds, unescaped keys and strings, parsed numbers, folded
  into a checksum by `walk.rs` (commutative over object entries, because `simd_json` objects
  iterate in hash order)

The `walk` checksums are compared across engines; a mismatch aborts the run.

| Engine | Parse | Walk | Threads |
|---|---|---|---|
| `simd` | `simd_json::to_borrowed_value` per row | DOM | 1 |
| `simd-par` | same, rayon over rows: how `genson-rs` runs | DOM | all |
| `simd-buf`, `simd-buf-par` | same, with reused `simd_json::Buffers` | DOM | 1 / all |
| `cujson-node` | `--tape`: `gpu` = `cujson::parse_lines`, `cpu` = `cujson::cpu::parse` | `Node` navigation | 1 |
| `cujson-node-par` | same | `Node` navigation, rayon over lines | all |
| `cujson-visit` | same | `Document::visit`, one pass over the tape | 1 |
| `cujson-visit-par` | same | `Document::visit_range` over `Document::split_lines` ranges in rayon | all |

Reported phases: simd-json's `copy` is the mutable copy of the batch it needs because it parses in
place; cuJSON's `gpu parse` (`cpu tape build` with `--tape cpu`) is one call covering host-to-device copy, kernels and the copy of the
tape back to pinned host memory; `walk` is CPU navigation of the tape; `free` is dropping the
document. `peak RSS` is the process high-water mark above the loaded corpus.

Not measured: parquet decode, the parts of `genson-rs` after parsing (schema strategies, regexes,
hash sets), and `polars`' `json_decode`.
