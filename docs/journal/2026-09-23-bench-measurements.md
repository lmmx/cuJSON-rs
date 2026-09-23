# 2026-09-23: cujson-bench measurements and experiments

Related entries: `2026-09-23-bench.md` (what is measured), `2026-09-23-tape-walk.md`, `2026-09-23-gpu-parse-overheads.md`.

Setup for every row: committed file (1,818 rows, 233.4 MB of JSON Lines), 20 CPU threads (Intel 10900K), RTX 3090 (sm_86, CUDA 13.2, PCIe link about 12 GB/s), `run --levels walk`, times are the median wall time of the whole file over 5 repetitions, cuJSON walk is `Document::visit_range` in rayon. Rows marked "container" ran on the driverless development container over a CPU-built tape.

## Current State

### Best result per configuration (walk level, seconds for the whole file)

| Commit | Configuration | `simd-par` | `cujson-visit-par` | `cujson-pipe` |
|---|---|---|---|---|
| 82b81aa-era (before ed01209) | 256 MB batch | 0.142 | 0.155 | not built |
| ed01209 | 256 MB batch | 0.142 | 0.154 | not built |
| ce7ae93 | 256 MB batch | 0.140 | 0.127 | not built |
| 63a7ea2 | 256 MB batch | 0.142 | 0.102 | 0.101 |
| 63a7ea2 | 32 MB batches | 0.110 | 0.113 | 0.080 |
| 5c2250b | 32 MB batches | 0.103 | 0.090 | 0.065 |
| 5c2250b + de94433 | 32 MB batches, `--pinned-input` | 0.102 | 0.085 | 0.061 |
| 8da77e4 | 32 MB, pinned, 1 GPU thread | | | 0.062 |
| fb4aab3 | 32 MB, pinned, 1 GPU thread | | | 0.054 |
| fb4aab3 | 32 MB, pinned, 2 GPU threads | | | 0.049 |
| fb4aab3 | 32 MB, pinned, 3 GPU threads | | | 0.052 |
| 6f08964 | 32 MB, pinned, 1 / 2 / 3 GPU threads (second run) | | | 0.054 / 0.052 / 0.050 |

- The first fb4aab3 rows ran `cujson-pipe` alone; whether the checksum reference added in 6f08964 was part of that build is not known, because 6f08964 was committed while that run was in flight; the second run (6f08964 row) started after 6f08964 was in the working tree and printed results for all three thread counts, so each passed the `simd-par` reference check
- Run-to-run spread at 32 MB, pinned input, `cujson-pipe`: 2 GPU threads 0.049 s and 0.052 s, 3 GPU threads 0.052 s and 0.050 s, 1 GPU thread 0.054 s in both runs, `[min-max]` ranges of 0.001 s to 0.005 s within a run

### Other files (`run --input <file> --batch-mb 32 --pinned-input --engines simd-par,cujson-pipe --levels walk`, master at d094037)

| File (pipeline copy) | Rows | Claims JSON | Mean row | Max row | Batches | `simd-par` | `cujson-pipe` | `parse busy` | `walk busy` |
|---|---|---|---|---|---|---|---|---|---|
| chunk_5-00037-of-00042 | 3,735 | 250.8 MB | 67,142 B | 1,280,862 B | 8 | 0.092 | 0.056 | 0.052 | 0.038 |
| chunk_10-00008-of-00017 | 4,200 | 239.9 MB | 57,109 B | 1,435,036 B | 8 | 0.098 | 0.064 | 0.058 | 0.041 |
| chunk_3-00031-of-00064 | 4,851 | 246.8 MB | 50,866 B | 520,078 B | 8 | 0.078 | 0.056 | 0.051 | 0.037 |

| chunk_0-00004-of-00546 | 1,818 | 1,648.9 MB | 906,987 B | 5,521,478 B | 51 | 1.232 | 0.370 | 0.357 | 0.353 |
| chunk_2-00114-of-00144 | 4,095 | 93.6 MB | 22,864 B | 840,191 B | 3 | 0.031 | 0.023 | 0.019 | 0.014 |

- `simd-par` reads 1.34 GB/s on chunk_0-00004 (`copy` 0.108 s, `parse+walk` 1.123 s, +539 MB RSS) and 3.07 GB/s on chunk_2-00114; `cujson-pipe` reads 4.45 GB/s (+588 MB RSS) and 4.05 GB/s (+162 MB RSS); ratios are 3.3 and 1.35
- `simd-par` throughput on the earlier three files reads 2.72, 2.45 and 3.18 GB/s (2.30 GB/s on chunk_0-00283); `cujson-pipe` reads 4.50, 3.73 and 4.42 GB/s (4.2 to 4.5 GB/s on chunk_0-00283); ratios of `simd-par` to `cujson-pipe` are 1.64, 1.53 and 1.39 against about 1.9 for chunk_0-00283
- Local pipeline copies range from 13 MB (`chunk_2-00114-of-00144`) to 751 MB (`chunk_0-00000-of-00546`) of parquet; the ten largest are all chunk_0 files of 488 MB to 751 MB


### Batch-size sweeps

- `simd-par` at 4, 8, 16, 32, 64, 128, 256 MB batches (commit 63a7ea2 era, then earlier for 4 and 8): 0.154, 0.146, 0.126 (0.123 on a later run), 0.110, 0.148, 0.147, 0.140 s, with the `copy` phase 0.018, 0.022, 0.019, 0.019, 0.066, 0.071, 0.068 s
- `cujson-visit-par` at ed01209 (before ce7ae93), at 4, 8, 16, 32, 64, 128, 256 MB: 0.316, 0.243, 0.202, 0.172, 0.165, 0.160, 0.155 s
- After 63a7ea2, `cujson-visit-par` at 16, 32, 64, 128, 256 MB: 0.137, 0.113, 0.113, 0.106, 0.102 s; `cujson-pipe` 0.085, 0.080, 0.082, 0.092, 0.101 s
- `gpu parse` of the whole file stays near 0.100 s for batches of 32 MB and larger before ce7ae93 (0.102, 0.102, 0.101, 0.098 s at 32, 64, 128, 256 MB)
- At 256 MB the file is one batch, so `cujson-pipe` cannot overlap and equals `cujson-visit-par` (0.101 s against 0.102 s)

### Single-thread comparison

- `simd` parse 0.44 s (0.502 s to 0.551 s with the `copy` phase included, depending on the run), first cuJSON `gpu parse` 0.090 s, a ratio of about 4.8 at the parse level

### Phase breakdown of `cujson-visit-par` (256 MB batch)

- Before ed01209: `gpu parse` 0.091 s to 0.104 s, `walk` 0.050 s, `free` 0.006 s to 0.010 s
- After ed01209: `gpu parse` 0.098 s, `walk` 0.050 s, `free` 0.006 s
- After ce7ae93: `gpu parse` 0.071 s, `walk` 0.050 s, `free` 0.006 s
- After 63a7ea2: `gpu parse` 0.052 s, `walk` 0.051 s, `free` 0.000 s
- After 5c2250b (32 MB batches, pageable input): `gpu parse` 0.052 s, `walk` 0.037 s; with `--pinned-input`: `gpu parse` 0.047 s

### Pipeline stage times (`cujson-pipe`, 32 MB batches, busy times summed over threads)

- 63a7ea2: `parse busy` 0.068 s, `walk busy` 0.069 s, wall 0.080 s
- 5c2250b: `parse busy` 0.059 s, `walk busy` 0.040 s, wall 0.065 s; with pinned input 0.056 s and 0.040 s, wall 0.061 s
- 78bbcf0, 1 / 2 / 3 GPU threads: `parse busy` 0.058 / 0.118 / 0.178 s, `walk busy` about 0.042 to 0.049 s, wall 0.064 / 0.069 / 0.073 s
- 8da77e4, 1 / 2 / 3 GPU threads: `parse busy` 0.056 / 0.097 / 0.140 s, `walk busy` 0.043 / 0.050 / 0.049 s, wall 0.062 / 0.062 / 0.062 s
- fb4aab3, 1 / 2 / 3 GPU threads: `parse busy` 0.049 / 0.079 / 0.111 s, `walk busy` 0.039 / 0.038 / 0.041 s, wall 0.054 / 0.049 / 0.052 s; second run: `parse busy` 0.049 / 0.082 / 0.110 s, `walk busy` 0.038 / 0.039 / 0.038 s, wall 0.054 / 0.052 / 0.050 s

### Peak RSS above the loaded corpus

- `simd-par` 196 MB to 411 MB depending on batch size (222 MB to 424 MB across the `simd*` engines at 256 MB); cuJSON engines 458 MB at 256 MB after ed01209 (918 MB to 4,790 MB before it), 309 MB to 329 MB at 32 MB batches after 63a7ea2, 41 MB to 63 MB and 168 MB in `cujson-pipe` runs at fb4aab3 depending on the run

## Experiments that did not change the result

- `--threads 18` and `--threads 19` on `cujson-pipe` at 32 MB batches (63a7ea2): wall 0.082 s and 0.084 s, against 0.080 s with 20 threads, with `parse busy` 0.071 s and 0.072 s
- `--gpu-threads 2` and `3` at 78bbcf0 (default thrust allocator): wall 0.069 s and 0.073 s against 0.064 s with one thread, with `parse busy` doubling and tripling; parses did not overlap
- `--pinned-input` lowered `gpu parse` by 5 ms (0.052 s to 0.047 s) where the copy time alone suggested about 12 ms; the host-to-device copy runs at about 11.6 GB/s from pinned memory and 9.5 GB/s from pageable memory
- `--gpu-threads 2` and `3` at 8da77e4 (asynchronous thrust allocator, one-buffer cache): wall 0.062 s for 1, 2 and 3 threads with `parse busy` rising to 0.097 s and 0.140 s
- A `split_lines` that scanned every tape entry serially: `cujson-visit-par` walk 0.126 s against 0.063 s for `cujson-node-par`

## Statements corrected during the work

- The `+MB RSS` column read before ed01209 included the leak; after ed01209 cuJSON's peak RSS is 458 MB at 256 MB batches against 372 MB to 394 MB for `simd-par`
- The tape is 79 MiB for the 233 MB input (0.35 times), not four times the input size as an early estimate from the leaking RSS figures suggested
- A first estimate that `--pinned-input` would save about 12 ms per file measured 5 ms
- A first estimate that walking the tape in one pass would remove most of the walk cost measured a 1.1 times gain (0.793 s to 0.714 s); string handling (`\u` decoding and per-string UTF-8 checks) accounted for most of the time

## Missing

- The same measurements on a file whose rows are much smaller
