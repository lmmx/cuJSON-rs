import ".just/ship.just"

default: check

# fmt, clippy (with CUDA), and tests that need no GPU
check:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets --features cujson-cli/cuda,cujson/cpu-reference,cujson/serde -- -D warnings
    cargo test --workspace

fmt:
    cargo fmt --all

# Every check against a real GPU (set CUJSON_CUDA_ARCHS=86 etc. to build faster)
verify:
    cargo gpu-verify

# Build the Python package into the active venv
[working-directory: 'crates/cujson-py']
py-dev:
    uv pip install -e . --group dev

# Python tests, including GPU tests
[working-directory: 'crates/cujson-py']
py-test *args:
    CUJSON_TEST_GPU=1 python -m pytest tests {{args}}

py: py-dev py-test

pc:
    pre-commit run --all-files
