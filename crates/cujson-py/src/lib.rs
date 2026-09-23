//! Python bindings for `cujson`, built with pyo3 (`abi3-py310`). The
//! compiled extension is `cujson._cujson`; `python/cujson/__init__.py`
//! re-exports its public names as `cujson.*` (task 08 brief).
//!
//! `Document` holds a `cujson::Document<'static>` built from
//! [`cujson::parse_owned`]/[`cujson::parse_lines_owned`], so it never
//! borrows a Python buffer across calls and is safe to hold across
//! threads (`#[pyclass(frozen)]`; `cujson::Document<'static>` is Send+Sync
//! because `TapeStorage` is, see `crates/cujson/src/tape/storage.rs`).
//! Every parse releases the GIL (`Python::detach`) for the actual GPU call.

mod convert;
mod exceptions;

use pyo3::PyTypeInfo;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes, PyDict, PyList, PyString};

fn extract_bytes(ob: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(s) = ob.cast::<PyString>() {
        return Ok(s.to_string().into_bytes());
    }
    if let Ok(b) = ob.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(b) = ob.cast::<PyByteArray>() {
        return Ok(unsafe { b.as_bytes() }.to_vec());
    }
    // memoryview and any other buffer-protocol object: go through `bytes(obj)`
    // rather than the (limited-API-gated) `PyBuffer` type.
    let as_bytes = pyo3::types::PyBytes::type_object(ob.py())
        .into_any()
        .call1((ob,))?;
    let b: &Bound<'_, PyBytes> = as_bytes.cast()?;
    Ok(b.as_bytes().to_vec())
}

/// A parsed document: owns its input bytes and GPU-produced tape.
#[pyclass(frozen)]
struct Document(cujson::Document<'static>);

#[pymethods]
impl Document {
    /// Full conversion to Python `dict`/`list`/`str`/`int`/`float`/`bool`/`None`.
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        convert::node_to_object(py, &self.0.root())
    }

    /// Resolve an RFC 6901 JSON Pointer, e.g. `"/0/user/lang"`.
    fn pointer(&self, py: Python<'_>, pointer: &str) -> PyResult<Py<PyAny>> {
        match self.0.pointer(pointer) {
            Some(node) => convert::node_to_object(py, &node),
            None => Err(PyKeyError::new_err(pointer.to_string())),
        }
    }

    /// Every top-level value, for a JSON Lines document (a single-element
    /// list for a standard document).
    fn lines(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: PyResult<Vec<Py<PyAny>>> = self
            .0
            .lines()
            .map(|node| convert::node_to_object(py, &node))
            .collect();
        Ok(PyList::new(py, items?)?.unbind())
    }

    fn __len__(&self) -> usize {
        self.0.lines().count()
    }

    fn __iter__(slf: PyRef<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = slf.lines(py)?;
        list.bind(py).call_method0("__iter__").map(|o| o.unbind())
    }
}

#[pyfunction]
fn parse(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Document> {
    let bytes = extract_bytes(data)?;
    py.detach(|| cujson::parse_owned(bytes))
        .map(Document)
        .map_err(|e| exceptions::to_pyerr(py, e))
}

#[pyfunction]
fn parse_file(py: Python<'_>, path: &str) -> PyResult<Document> {
    let path = path.to_string();
    py.detach(|| cujson::parse_file(path))
        .map(Document)
        .map_err(|e| exceptions::to_pyerr(py, e))
}

#[pyfunction]
#[pyo3(signature = (data, chunk_bytes=None))]
fn parse_lines(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    chunk_bytes: Option<usize>,
) -> PyResult<Document> {
    let bytes = extract_bytes(data)?;
    let opts = cujson::LinesOptions {
        chunk_bytes: chunk_bytes.unwrap_or_else(|| cujson::LinesOptions::default().chunk_bytes),
    };
    py.detach(|| cujson::parse_lines_owned(bytes, opts))
        .map(Document)
        .map_err(|e| exceptions::to_pyerr(py, e))
}

/// CUDA runtime/device info. Built without the `cuda` feature, this
/// returns `{"compiled": False}` rather than raising — `cuda_info()`'s own
/// job is to report on availability, not require it. With `cuda` compiled
/// in but no driver (or no device) present, it raises `CudaError`/
/// `NoDeviceError` — a clean exception, not a crash, which is the
/// behaviour a driverless container exercises for real (this crate's
/// journal entry marks that unverified-on-a-real-GPU, tier 3).
#[pyfunction]
fn cuda_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    match cujson::cuda_info() {
        Ok(info) => {
            let dict = PyDict::new(py);
            dict.set_item("compiled", true)?;
            dict.set_item("runtime_version", info.runtime_version)?;
            dict.set_item("compiled_archs", info.compiled_archs)?;
            let devices = PyList::empty(py);
            for d in info.devices {
                let dev = PyDict::new(py);
                dev.set_item("index", d.index)?;
                dev.set_item("name", d.name)?;
                devices.append(dev)?;
            }
            dict.set_item("devices", devices)?;
            Ok(dict.into_any().unbind())
        }
        Err(cujson::Error::CudaNotCompiled) => {
            let dict = PyDict::new(py);
            dict.set_item("compiled", false)?;
            Ok(dict.into_any().unbind())
        }
        Err(e) => Err(exceptions::to_pyerr(py, e)),
    }
}

#[pymodule]
fn _cujson(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    exceptions::init(py, module)?;
    module.add_class::<Document>()?;
    module.add_function(wrap_pyfunction!(parse, module)?)?;
    module.add_function(wrap_pyfunction!(parse_file, module)?)?;
    module.add_function(wrap_pyfunction!(parse_lines, module)?)?;
    module.add_function(wrap_pyfunction!(cuda_info, module)?)?;
    Ok(())
}
