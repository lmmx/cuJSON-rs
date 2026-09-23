# 2026-09-23: CUDA-only Python Package and Single-Value Check

## Current State

- `cujson-py` has no `cuda` feature and depends on `cujson` with `features = ["cuda", "serde"]` (crates/cujson-py/Cargo.toml) — `CUDA_COMPILED`, `CudaNotCompiledError` and the `{"compiled": False}` branch of `cuda_info()` are removed; `pip install -e crates/cujson-py` builds the GPU extension (commit bc80af6)
- `release.yml`'s `cli` and `wheel` jobs build CUDA 13 only — the Python distribution is `cujson`, the CLI tarball `cujson-cu13-x86_64-unknown-linux-gnu.tar.gz`, with no CUDA 12 artifacts; README.md directs CUDA 12 users to build from a checkout (`pip install ./crates/cujson-py` built and passed pytest here with nvcc 12.8, tier 2)
- `ci.yml`'s `cuda-compile` job still covers CUDA 12.8.1 and 13.0.0
- `ci.yml`'s `check` job lints and tests the workspace with `--exclude cujson-py`; the `cuda-compile` job installs uv (`astral-sh/setup-uv@v6`), builds the wheel with `uv pip install ./crates/cujson-py pytest`, and runs pytest — mirrored in this container: 5 passed, 4 GPU tests skipped (tier 2); `actionlint` reports no findings; neither workflow has run on GitHub
- `Document::is_single_value()` (crates/cujson/src/tape/document.rs) accepts one bracketed root whose opener is tape index 1 and whose `pair_pos` is `len - 2` with only whitespace outside it, or, for a tape with no structural entries, one scalar token — `cujson::parse`, `parse_owned` and `cpu::parse(.., Mode::Standard)` return `NotSingleValue` otherwise, raised as `InputError` in Python (commit 6223606)
- Before 6223606, `cujson.parse('{"hello": "world"}\n{"bonjour":"monde"}')` on the user's RTX 3090 returned a Document whose `to_python()` was the first object and whose `lines()` was `[{'hello': 'world'}, 'bonjour', 'monde']`; the CPU reference also accepted `1 2` as `null`
- `standard_mode_accepts_exactly_one_value` (crates/cujson/tests/tape_tests.rs) covers 7 accepted and 7 rejected inputs through the CPU reference (tier 1); `verify` adds "error recovery: standard two values -> NotSingleValue" and pytest adds `test_gpu_parse_rejects_json_lines` — both GPU-only and not yet run (tier 3)

## Missing

- A top-level scalar with internal structure the tape cannot see (e.g. a malformed number `1.2.3`) passes `is_single_value()`; cuJSON does not validate scalar grammar
