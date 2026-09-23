# 01 — Workspace scaffold and pristine upstream import

Depends on: nothing. Verification tier: 1.

## Deliverables

- Root `Cargo.toml` workspace with `members = ["crates/*"]`, `resolver = "3"`, shared `[workspace.package]` (edition, rust-version, license MIT, repository) and `[workspace.dependencies]`
- `rust-toolchain.toml` pinning stable
- Empty crates that build without CUDA:
  - `crates/cujson-sys` (lib, `links = "cujson"`, features `cuda = ["dep:cudaforge"]`, `build.rs` with a no-op main)
  - `crates/cujson` (lib, depends on `cujson-sys`, features `cuda`, `cpu-reference`, `serde`)
  - `crates/cujson-cli` (bin named `cujson`)
  - `crates/cujson-py` is **not** created here (task 08)
- Pristine copy of `vendor/cuJSON/cujson/` → `crates/cujson-sys/cuda/upstream/`, byte-identical, in its own commit
- `crates/cujson-sys/cuda/upstream/LICENSE` (copied from `vendor/cuJSON/LICENSE`) and `UPSTREAM.md` naming repo URL, commit `38d27b6e6c4eb74205cf59f4123b0983034405e2`, and the rule that local changes land as separate commits
- Test fixtures: copy `vendor/cuJSON/dataset/twitter_sample_large_record.json` and `twitter_sample_small_records.json` to `tests/fixtures/` at the workspace root, with a `tests/fixtures/README.md` noting their origin
- `.gitignore` gains `target/`

## Acceptance

- `cargo build --workspace`, `cargo clippy --workspace -- -D warnings`, `cargo test --workspace` pass with no CUDA toolkit
- `diff -r vendor/cuJSON/cujson crates/cujson-sys/cuda/upstream` shows only the added LICENSE/UPSTREAM.md
- `git log` shows the import as a standalone commit
