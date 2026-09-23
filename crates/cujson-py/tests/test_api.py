"""pytest tests for the cujson Python package (task 08 brief).

Non-GPU tests run at tier 1 (any wheel, CUDA or not). GPU tests (marked
``gpu``) are skipped unless ``CUJSON_TEST_GPU=1`` and compare against
``json.loads`` on both fixtures in ``tests/fixtures`` at the repo root.
"""

import json
import os
import pathlib

import pytest

import cujson

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests" / "fixtures"
LARGE_RECORD = FIXTURES / "twitter_sample_large_record.json"
SMALL_RECORDS = FIXTURES / "twitter_sample_small_records.json"

GPU_ENABLED = os.environ.get("CUJSON_TEST_GPU") == "1"


def skip_no_gpu(func):
    """@pytest.mark.gpu, skipped unless CUJSON_TEST_GPU=1."""
    func = pytest.mark.skipif(
        not GPU_ENABLED, reason="set CUJSON_TEST_GPU=1 to run GPU tests"
    )(func)
    return pytest.mark.gpu(func)


def test_import():
    assert hasattr(cujson, "parse")
    assert hasattr(cujson, "parse_file")
    assert hasattr(cujson, "parse_lines")
    assert hasattr(cujson, "cuda_info")


def test_exception_hierarchy():
    assert issubclass(cujson.InvalidUtf8Error, cujson.CujsonError)
    assert issubclass(cujson.InvalidUtf8Error, ValueError)
    assert issubclass(cujson.UnbalancedError, cujson.CujsonError)
    assert issubclass(cujson.UnbalancedError, ValueError)
    assert issubclass(cujson.CudaError, cujson.CujsonError)
    assert issubclass(cujson.CudaError, RuntimeError)
    assert issubclass(cujson.CudaNotCompiledError, cujson.CudaError)
    assert issubclass(cujson.NoDeviceError, cujson.CudaError)
    assert issubclass(cujson.InputError, cujson.CujsonError)
    assert issubclass(cujson.InputError, ValueError)


def test_every_extension_export_is_reexported():
    from cujson import _cujson

    public = {name for name in dir(_cujson) if not name.startswith("_")}
    assert public <= set(cujson.__all__)
    assert all(hasattr(cujson, name) for name in public)


def test_empty_input_raises_input_error():
    # Checked before any CUDA call, so this holds for CUDA and non-CUDA wheels.
    with pytest.raises(cujson.InputError):
        cujson.parse(b"")


def test_cuda_info_reports_a_dict():
    info = cujson.cuda_info()
    assert isinstance(info, dict)
    assert "compiled" in info
    assert isinstance(info["compiled"], bool)


@pytest.mark.skipif(GPU_ENABLED, reason="only meaningful for a non-CUDA build")
def test_cuda_info_not_compiled_reports_false():
    info = cujson.cuda_info()
    if info["compiled"] is True:
        pytest.skip("this wheel was built with the cuda feature")
    assert info["compiled"] is False


def test_parse_without_cuda_raises_helpful_error():
    """On a non-CUDA build, parse() must raise CudaNotCompiledError naming
    which wheel to install instead. On a CUDA build with no driver, this
    instead raises CudaError/NoDeviceError (tier 3, see gpu tests)."""
    info = cujson.cuda_info()
    if info["compiled"]:
        pytest.skip("this wheel was built with the cuda feature")
    with pytest.raises(cujson.CudaNotCompiledError) as exc_info:
        cujson.parse(b'{"a": 1}')
    message = str(exc_info.value)
    assert "cujson-cu12" in message
    assert "cujson-cu13" in message


@skip_no_gpu
def test_gpu_parse_matches_json_loads_small_records():
    text = SMALL_RECORDS.read_text()
    lines = [json.loads(line) for line in text.splitlines() if line.strip()]
    doc = cujson.parse_lines(text.encode("utf-8"))
    assert doc.lines() == lines
    assert len(doc) == len(lines)
    assert doc.pointer("/0/id") == lines[0]["id"]


@skip_no_gpu
def test_gpu_parse_matches_json_loads_large_record():
    text = LARGE_RECORD.read_text()
    expected = json.loads(text)
    doc = cujson.parse(text.encode("utf-8"))
    assert doc.to_python() == expected


@skip_no_gpu
def test_gpu_parse_file():
    expected = json.loads(LARGE_RECORD.read_text())
    doc = cujson.parse_file(str(LARGE_RECORD))
    assert doc.to_python() == expected


@skip_no_gpu
def test_gpu_cuda_info_reports_compiled_and_devices():
    info = cujson.cuda_info()
    assert info["compiled"] is True
    assert len(info["devices"]) >= 1
