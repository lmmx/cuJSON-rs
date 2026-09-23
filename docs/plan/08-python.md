# 08 — Python bindings (`cujson-py`)

Depends on: 06. Verification tier: 1 for building a non-CUDA wheel and running pytest against it; tier 3 for GPU behaviour.

Reference: `/mnt/mistral.rs/mistralrs-pyo3/` (Cargo.toml, pyproject.toml).

## Layout

- `crates/cujson-py/Cargo.toml`: `cdylib` named `cujson`, `publish = false`, `pyo3` with `abi3-py310` and `extension-module`, feature `cuda = ["cujson/cuda"]`
- `crates/cujson-py/pyproject.toml`: maturin build backend, `module-name = "cujson._cujson"`, a python-source dir with `cujson/__init__.py` re-exporting, and a `.pyi` stub plus `py.typed`
- The distribution name gets rewritten at release time (task 09): `cujson-cu12` and `cujson-cu13`, both providing import name `cujson`. In-repo `name = "cujson"`

## API

```python
import cujson
cujson.cuda_info() -> dict
doc = cujson.parse(data: bytes | bytearray | memoryview | str)   # str is UTF-8-encoded
doc = cujson.parse_file(path)
doc = cujson.parse_lines(data, chunk_bytes=...)
doc.pointer("/0/user/lang")          # -> Python object, or raises KeyError
doc.to_python()                      # full conversion to dict/list/...
len(doc.lines()) / iterate lines -> each yields a Python object
```

- `Document` holds `cujson::Document<'static>` (built with `parse_owned`), so no Python buffer is borrowed across calls
- Release the GIL (`py.allow_threads`) around the GPU parse
- Exceptions: `cujson.CujsonError` base; `ValueError` subclasses for invalid UTF-8 and unbalanced input; `RuntimeError` subclass for CUDA / not-compiled errors, with the message naming `cujson info`-equivalent diagnostics (`cujson.cuda_info()`)

## Tests

- `crates/cujson-py/tests/test_api.py` (pytest): import works; without CUDA, `parse` raises the not-compiled error with a helpful message; `cuda_info()` reports `compiled: False`
- GPU tests get marked `@pytest.mark.gpu` and skipped unless `CUJSON_TEST_GPU=1`: fixture parse equals `json.loads`

## Acceptance

- Tier 1: `maturin build -m crates/cujson-py/Cargo.toml` (install maturin into a scratch venv with `uv` or `pip`), install the wheel into a venv, and pytest passes
