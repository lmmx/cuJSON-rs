# 2026-09-23: `gpu-stage-trim` experiments (branch not merged)

Related entries: `2026-09-23-gpu-parse-overheads.md`, `2026-09-23-tape-walk.md`, `2026-09-23-bench-measurements.md`.

## Current State

- Branch `gpu-stage-trim` (commits `87ebce7` to `67decdd`, eight commits on top of `d094037`) holds four experiments on the GPU stage and the walk of `cujson-pipe`; the branch was not merged and `master` does not contain any of its code
- Measurements below are from the RTX 3090 host, `run --batch-mb 32 --pinned-input --engines cujson-pipe --levels walk`, whole-file median wall time; `master` reads 0.054 s with `parse busy` 0.049 s and `walk busy` 0.038 s to 0.040 s in every run of every A/B (four runs, +165 MB RSS)

### Experiment 1: `split_lines` without `pair_pos` (`87ebce7`, `8bb581a`)

- `87ebce7` replaced the hop through `pair_pos` at line openers with a forward search for a `\n` tape entry from each of `threads * 16` even offsets, run serially before the parallel walk
- On the CPU-built tape walked as one 233 MB batch it cost 3 ms; in `cujson-pipe` the search ran once per 32 MB batch on the consumer thread and `walk busy` rose from 0.038 s to 0.057 s to 0.059 s (`cujson-pipe` 0.054 s to 0.064 s); the same jump reproduced without a GPU (CPU tape, `--batch-mb 32`, `walk` phase 0.039 s at `master`, 0.058 s at `87ebce7`)
- `8bb581a` (`Document::line_part`) moved each boundary search into the parallel loop; on the CPU tape the phase read 0.040 s to 0.041 s, and on the host `walk busy` read 0.041 s to 0.042 s against 0.038 s to 0.040 s at `master`
- The newline search needs no `pair_pos`; an empty part returns `n..n`, and an intermediate version visited a trailing scalar twice (found by `lines_edge_cases`, input `1\n2\n3`)

### Experiment 2: tape without `pair_pos` (`e9826fd`)

- `cujson_parse_lines_ex` with `CUJSON_FLAG_NO_PAIR_POS` and `cujson::parse_lines_without_pair_pos` skipped the `pair_pos` device-to-host copy and allocated `N + 2` instead of `2N + 3` tape entries; `Document::has_pair_pos()` reported it and `Node` navigation panicked on such a document
- `--no-pair-pos` runs (regression fixed): `cujson-pipe` 0.052 s, `parse busy` 0.046 s, +136 MB RSS, against 0.055 s, 0.050 s and +150 MB for the same branch without the option
- The two A/B runs of the branch without `--no-pair-pos` read 0.055 s to 0.056 s, `walk busy` 0.041 s to 0.042 s and +146 MB to +152 MB RSS, against 0.054 s and 0.038 s to 0.040 s at `master`
- `parse_without_pair_pos_matches_the_full_parse` passed on the host at three chunk sizes

### Experiment 3: batches that start and end small (`855bda7`, removed in `67decdd`)

- `run --ramp` cut the corpus into 11 batches (1/8, 1/4, 1/2, full-size, 1/2, 1/4, 1/8 of `--batch-mb`); results are in `2026-09-23-bench-measurements.md` under "Experiments that did not change the result": 0.056 s and 0.053 s (with `--no-pair-pos`), against 0.055 s and 0.052 s for uniform batches

### Experiment 4: pinned cache counters, headroom and reuse bound (`c0d0594`, `69ba010`)

- `cujson::pinned_cache_stats()` counted 47 hits, 2 `cudaMallocHost` and 0 `cudaFreeHost` per uniform-batch run; a request could reuse a cached buffer up to 2x its size, later 8x, and new buffers got 25% headroom
- With the 11-batch ramp the counters read 31 `cudaMallocHost` and 29 `cudaFreeHost` at a 2x bound and 5 and 3 at 8x; the uniform-batch counters did not change with either rule

## Outcome

- The A/B of `gpu-stage-trim` against `master` at default options showed `cujson-pipe` 0.055 s to 0.056 s against 0.054 s and `walk busy` about 2 ms higher; the only configuration faster than `master` was `--no-pair-pos` at 0.052 s
- The branch was set aside rather than merged; the persistent device workspace for the input buffer's `cudaMalloc` was not built (the profile after the async thrust allocator showed about 1 ms of `cudaMalloc` per 233 MB pass)
