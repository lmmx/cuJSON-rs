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
#[pyclass(frozen, module = "cujson")]
struct Document {
    doc: cujson::Document<'static>,
    /// Built by `parse_lines`: the document is the sequence of its lines.
    is_lines: bool,
}

#[pymethods]
impl Document {
    /// Full conversion to Python `dict`/`list`/`str`/`int`/`float`/`bool`/`None`.
    /// A JSON Lines document converts to the list of its lines.
    fn to_python(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if self.is_lines {
            return Ok(self.lines(py)?.into_any());
        }
        convert::node_to_object(py, &self.doc.root())
    }

    /// Resolve an RFC 6901 JSON Pointer, e.g. `"/0/user/lang"`. For a JSON
    /// Lines document the first token selects the line, so `"/1/user"` is
    /// line 1's `user` and `""` is every line.
    fn pointer(&self, py: Python<'_>, pointer: &str) -> PyResult<Py<PyAny>> {
        let not_found = || PyKeyError::new_err(pointer.to_string());
        if !self.is_lines {
            let node = self.doc.pointer(pointer).ok_or_else(not_found)?;
            return convert::node_to_object(py, &node);
        }
        if pointer.is_empty() {
            return self.to_python(py);
        }
        let rest = pointer.strip_prefix('/').ok_or_else(not_found)?;
        let (line, rest) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        let index: usize = line.parse().map_err(|_| not_found())?;
        let node = self
            .doc
            .lines()
            .nth(index)
            .and_then(|line| line.pointer(rest))
            .ok_or_else(not_found)?;
        convert::node_to_object(py, &node)
    }

    /// Every top-level value, for a JSON Lines document (a single-element
    /// list for a standard document).
    fn lines(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: PyResult<Vec<Py<PyAny>>> = self
            .doc
            .lines()
            .map(|node| convert::node_to_object(py, &node))
            .collect();
        Ok(PyList::new(py, items?)?.unbind())
    }

    fn __len__(&self) -> usize {
        self.doc.lines().count()
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
        .map(|doc| Document {
            doc,
            is_lines: false,
        })
        .map_err(|e| exceptions::to_pyerr(py, e))
}

#[pyfunction]
fn parse_file(py: Python<'_>, path: &str) -> PyResult<Document> {
    let path = path.to_string();
    py.detach(|| cujson::parse_file(path))
        .map(|doc| Document {
            doc,
            is_lines: false,
        })
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
        .map(|doc| Document {
            doc,
            is_lines: true,
        })
        .map_err(|e| exceptions::to_pyerr(py, e))
}

/// CUDA runtime/device info. Raises `CudaError`/`NoDeviceError` when no
/// usable driver or device is present.
#[pyfunction]
fn cuda_info(py: Python<'_>) -> PyResult<Py<PyAny>> {
    match cujson::cuda_info() {
        Ok(info) => {
            let dict = PyDict::new(py);
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
        Err(e) => Err(exceptions::to_pyerr(py, e)),
    }
}

/// Release the pinned host buffer kept between parses (at most one is kept,
/// so the next parse can skip allocating). Safe to call at any time, with
/// or without a GPU.
#[pyfunction]
fn trim_pinned_cache() {
    cujson::trim_pinned_cache();
}

#[pymodule]
fn _cujson(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    exceptions::init(py, module)?;
    module.add_class::<Document>()?;
    module.add_function(wrap_pyfunction!(parse, module)?)?;
    module.add_function(wrap_pyfunction!(parse_file, module)?)?;
    module.add_function(wrap_pyfunction!(parse_lines, module)?)?;
    module.add_function(wrap_pyfunction!(cuda_info, module)?)?;
    module.add_function(wrap_pyfunction!(trim_pinned_cache, module)?)?;
    Ok(())
}
