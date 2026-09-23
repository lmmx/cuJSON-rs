# 09 — CI and release workflows

Depends on: 04 for the CI compile check; 07 and 08 for release artifacts. Verification tier: workflows are YAML the agent can't run. Validate with `actionlint` if installable, and have the user run with `workflow_dispatch` + `dry_run`.

Reference: `/mnt/mistral.rs/.github/workflows/release.yml` (linux-cuda job ≈302-490) and `ci.yml`.

## `.github/workflows/ci.yml` (push, PR)

- `check` (ubuntu-latest): fmt, clippy `-D warnings`, `cargo test --workspace` (no CUDA), and a maturin non-CUDA wheel build plus pytest
- `cuda-compile` matrix `{cuda: ["12.8.1", "13.0.0"]}`, `container: nvidia/cuda:${{ matrix.cuda }}-devel-ubuntu22.04`: install rustup; `cargo clippy --workspace --features cuda -- -D warnings`; `cargo build --workspace --features cuda`; `cargo test --workspace --features cuda` (GPU tests are `#[ignore]`d, so this proves link and startup; the non-ignored tests must not touch the GPU); then an `ldd` check on the CLI binary failing if `libcudart` shows up (static cudart per README decision). Set `CUJSON_CUDA_ARCHS=80` for speed on PRs. One job without the override proves the full arch list compiles

This job carries tier 2 for tasks 02-04 and 06.

## `.github/workflows/release.yml` (tag `v*`, plus `workflow_dispatch` with `dry_run`)

Matrix `{cuda_major: 12 → image 12.8.1, 13 → image 13.0.0}` × x86_64 linux:
- CLI: `cargo build --release -p cujson-cli --features cuda`, then tar as `cujson-cu{12,13}-x86_64-unknown-linux-gnu.tar.gz`. No lib bundling needed (static cudart); assert with `ldd`
- Wheel: build in a manylinux_2_28 container with the CUDA toolkit installed from NVIDIA's RHEL8 repo (the nvidia/cuda Ubuntu images yield only manylinux_2_35 wheels). Rewrite `name = "cujson"` → `cujson-cu12`/`cujson-cu13` in pyproject.toml, as mistral.rs does for the version, then `maturin build --release --features cuda`. `auditwheel show` must report no external CUDA libs. Use the ubuntu22.04 image + `auditwheel repair` fallback only if the manylinux route fails, and record why
- Upload: GitHub release assets for both (via `softprops/action-gh-release`), PyPI via `pypa/gh-action-pypi-publish` with trusted publishing in a `pypi` environment, and crates.io publish of `cujson-sys` then `cujson`, gated on non-dry-run tags. **The agent writes these steps but the user configures tokens/trusted publishers; nothing gets published during this task**
- sdist: publish none for the `-cuXX` packages (a source build needs nvcc; point such users to the crate instead)

`cargo package -p cujson-sys --list` must include `cuda/**` and exclude nothing needed. Check the crate size is under crates.io's 10 MB limit.

## Acceptance

- Workflows exist and pass `actionlint` (if installable); `cargo package -p cujson-sys --allow-dirty --no-verify` succeeds and lists the CUDA sources
- Journal entry lists which jobs have actually run (none, until the user triggers them)
