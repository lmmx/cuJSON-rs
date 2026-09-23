# 2026-09-23: CUDA-only Python Package and Single-Value Check

## Current State

- `cujson-py` has no `cuda` feature and depends on `cujson` with `features = ["cuda", "serde"]` (crates/cujson-py/Cargo.toml) — `CUDA_COMPILED`, `CudaNotCompiledError` and the `{"compiled": False}` branch of `cuda_info()` are removed; `pip install -e crates/cujson-py` builds the GPU extension (commit bc80af6)
- `release.yml`'s `cli` and `wheel` jobs build CUDA 13 only — the Python distribution is `cujson`, the CLI tarball `cujson-cu13-x86_64-unknown-linux-gnu.tar.gz`, with no CUDA 12 artifacts; README.md directs CUDA 12 users to build from a checkout (`pip install ./crates/cujson-py` built and passed pytest here with nvcc 12.8, tier 2)
- `ci.yml`'s `cuda-compile` job still covers CUDA 12.8.1 and 13.0.0
- `ci.yml`'s `check` job lints and tests the workspace with `--exclude cujson-py`; the `cuda-compile` job installs uv (`astral-sh/setup-uv@v6`), builds the wheel with `uv pip install ./crates/cujson-py pytest`, and runs pytest — mirrored in this container: 5 passed, 4 GPU tests skipped (tier 2); `actionlint` reports no findings; neither workflow has run on GitHub
- `Document::is_single_value()` (crates/cujson/src/tape/document.rs) accepts one bracketed root whose opener is tape index 1 and whose `pair_pos` is `len - 2` with only whitespace outside it, or, for a tape with no structural entries, one scalar token — `cujson::parse`, `parse_owned` and `cpu::parse(.., Mode::Standard)` return `NotSingleValue` otherwise, raised as `InputError` in Python (commit 6223606)
- Before 6223606, `cujson.parse('{"hello": "world"}\n{"bonjour":"monde"}')` on the user's RTX 3090 returned a Document whose `to_python()` was the first object and whose `lines()` was `[{'hello': 'world'}, 'bonjour', 'monde']`; the CPU reference also accepted `1 2` as `null`
- `standard_mode_accepts_exactly_one_value` (crates/cujson/tests/tape_tests.rs) covers 7 accepted and 7 rejected inputs through the CPU reference (tier 1); `verify` adds "error recovery: standard two values -> NotSingleValue" and pytest adds `test_gpu_parse_rejects_json_lines` — both GPU-only and not yet run (tier 3)

- The Python distribution is `cujson` (crates/cujson-py/pyproject.toml), matching the `cujson` and `cujson-sys` crates the user published to crates.io at 0.1.0 — it was briefly `cujson-rs` (commit 7847bda) before those crates were published
- `release.yml`'s `crates-publish` job obtains its crates.io token from `rust-lang/crates-io-auth-action@v1` with `id-token: write` instead of a `CARGO_REGISTRY_TOKEN` secret; no workflow references a repository secret
- README.md documents a manual first release (`cargo publish` of `cujson-sys` then `cujson`; `uv publish` of the `wheel-cu13` artifact from a dry-run `release.yml`) followed by trusted-publisher setup on crates.io and PyPI, and `just release` (.just/ship.just) for later releases — `just release patch` was run against a scratch clone with a local bare `origin` and produced the bump commit and `v0.1.1` tag there
- The first scratch attempt at that test pushed a "wip justfile" commit to the user's local `master` in /mnt/cuJSON-rs (its `origin` had not yet been repointed); `master` was reset to `d330662` (= `origin/master`) and nothing reached GitHub

## Missing

- A top-level scalar with internal structure the tape cannot see (e.g. a malformed number `1.2.3`) passes `is_single_value()`; cuJSON does not validate scalar grammar
