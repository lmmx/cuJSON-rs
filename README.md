# cuJSON-rs

[cuJSON](https://github.com/AutomataLab/cuJSON) (ASPLOS '26) parses JSON and JSON Lines on an
NVIDIA GPU. cuJSON-rs packages its CUDA kernels so they build with `cargo` and install with
`pip`, with no manual `nvcc` step.

| Package | Install | Docs |
|---|---|---|
| Python | `pip install cujson` | [crates/cujson-py](crates/cujson-py/README.md) |
| Rust | `cargo add cujson --features cuda` | [crates/cujson](crates/cujson/README.md) |
| Raw FFI | `cargo add cujson-sys` | [crates/cujson-sys](crates/cujson-sys/README.md) |
| CLI | `cargo install --path crates/cujson-cli --features cuda` | `cujson --help` |

The Python wheel targets Linux x86_64 with an NVIDIA driver supporting CUDA 13 (R580 or newer)
and a GPU of compute capability 7.5 or newer. Building from source needs the CUDA toolkit (12.1
or newer).

To check everything against your GPU from a checkout:

```
CUJSON_CUDA_ARCHS=86 cargo gpu-verify   # set to your GPU's compute capability, or omit to build all
```

The kernels are upstream cuJSON's, copied from commit `38d27b6` and patched to be usable as a
library (see [crates/cujson-sys](crates/cujson-sys/README.md) and `docs/journal/`). cuJSON is by
Ashkan Vedadi Gargary, Soroosh Safari Loaliyan and Zhijia Zhao; cite
[CuJSON: A Highly Parallel JSON Parser for GPUs](https://doi.org/10.1145/3760250.3762222) if you
use it in research. Both projects are MIT licensed.

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
  `crates-publish` (crates.io trusted publishing via `rust-lang/crates-io-auth-action`) run only when
  `github.event_name == 'push' && startsWith(github.ref, 'refs/tags/v') && !inputs.dry_run` —
  a manual dispatch never publishes.

Release artifacts: the `cujson` Python wheel and a `cujson-cu13-*.tar.gz` CLI tarball, both
built with CUDA 13, and the `cujson-sys`/`cujson` crates on crates.io. On CUDA 12, build from a
checkout instead: `pip install ./crates/cujson-py` or `cargo build --release -p cujson-cli
--features cuda` (with CUDA 12's `nvcc` on `PATH`).

### Making a release

One version, `[workspace.package] version` in the root `Cargo.toml`, covers every crate and the
Python wheel (maturin reads it). The Python package shares its name, `cujson`, with the main crate.

**Prerequisite (one-time):** trusted publishing must be configured before the first tagged
release, or its publish jobs fail. Point each at repository `lmmx/cuJSON-rs`, workflow
`release.yml`:

- PyPI project `cujson`: Manage → Publishing → add a GitHub publisher with environment `pypi`,
  and create a `pypi` environment in this repository's Settings → Environments.
- crates.io crates `cujson` and `cujson-sys`: Settings → Trusted Publishing.

Version 0.1.0 was published by hand (no `v0.1.0` tag exists, deliberately).

Then, from an up-to-date, clean `master`:

```
just release          # or: just release minor / just release major
```

This runs `just check`, bumps the version with `cargo set-version`, commits
`chore(release): bump -> vX.Y.Z`, tags `vX.Y.Z`, and pushes both. The tag starts `release.yml`,
which publishes to PyPI and crates.io through trusted publishing (no stored tokens) and creates
a GitHub release. Local tools: `just`, `cargo set-version` (cargo-edit), `jq`, `echo-comment`.
