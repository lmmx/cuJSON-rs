# cuJSON-rs
Rust implementation of cuJSON: A Highly Parallel JSON Parser for GPUs (ASPLOS ‘26)

## Building and releasing

`.github/workflows/ci.yml` runs on every push and PR:

- `check`: `cargo fmt --check`, `cargo clippy` and `cargo test` over the workspace except
  `cujson-py`, which always builds with CUDA (no CUDA toolkit on this runner).
- `cuda-compile`: a `{cuda: ["12.8.1", "13.0.0"]}` matrix in `nvidia/cuda:*-devel-ubuntu22.04`
  containers. Runs `cargo clippy`/`build`/`test --features cuda` (GPU-only tests are
  `#[ignore]`d, so this proves link and startup without a GPU), then asserts with `ldd` that
  the built `cujson` CLI binary has no dynamic `libcudart` dependency (cudart is linked
  statically), then builds the Python wheel and runs `pytest` against it. One matrix leg pins `CUJSON_CUDA_ARCHS=80` for speed; the other builds the full
  default arch list.

`.github/workflows/release.yml` runs on `v*.*.*` tags and on manual `workflow_dispatch` (with a
`dry_run` input, default `true`, that builds everything but publishes nothing):

- `cli`: builds `cujson-cli --features cuda` with CUDA 13 and packages
  `cujson-cu13-x86_64-unknown-linux-gnu.tar.gz`, asserting no dynamic `libcudart` via `ldd`.
- `wheel`: builds the CUDA 13 Python wheel inside a `manylinux_2_28` container with the
  CUDA toolkit installed from NVIDIA's RHEL8 package repo (the `nvidia/cuda` Ubuntu images only
  yield `manylinux_2_35` wheels), and
  checking with `auditwheel show` that no CUDA shared library other than `libcuda` (which is
  dlopen'd by the driver at runtime, never bundled) appears in the wheel.
- `crate-package-check`: lists and size-checks the `cujson-sys` `.crate` (crates.io's 10 MB
  limit).
- `github-release`, `pypi-publish` (PyPI trusted publishing via a `pypi` environment), and
  `crates-publish` (crates.io via a `CARGO_REGISTRY_TOKEN` secret) run only when
  `github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v') && !inputs.dry_run` —
  a manual dispatch never publishes.

Release artifacts: the `cujson` Python wheel and a `cujson-cu13-*.tar.gz` CLI tarball, both
built with CUDA 13, and the `cujson-sys`/`cujson` crates on crates.io. On CUDA 12, build from a
checkout instead: `pip install ./crates/cujson-py` or `cargo build --release -p cujson-cli
--features cuda` (with CUDA 12's `nvcc` on `PATH`).
