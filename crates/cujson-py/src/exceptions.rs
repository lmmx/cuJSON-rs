//! Exception hierarchy, built at module-init time via Python's own `type()`
//! builtin so each leaf class can have two bases (`CujsonError` plus a
//! matching builtin) — `pyo3::create_exception!` only supports one base.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyTuple, PyType};

static CUJSON_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static INVALID_UTF8_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static UNBALANCED_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static INPUT_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static CUDA_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static NO_DEVICE_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();

fn make_type<'py>(
    py: Python<'py>,
    name: &str,
    bases: &Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyType>> {
    let builtins = PyModule::import(py, "builtins")?;
    let type_fn = builtins.getattr("type")?;
    let dict = PyDict::new(py);
    dict.set_item("__module__", "cujson")?;
    let cls = type_fn.call1((name, bases, dict))?;
    cls.cast_into::<PyType>()
        .map_err(|e| pyo3::PyErr::from_value(e.into_inner()))
}

/// Build the hierarchy and register every class on `module`. Called once
/// from the `#[pymodule]` entry point.
pub fn init(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    let exc = py.get_type::<pyo3::exceptions::PyException>();
    let value_error = py.get_type::<pyo3::exceptions::PyValueError>();
    let runtime_error = py.get_type::<pyo3::exceptions::PyRuntimeError>();

    let cujson_error = make_type(py, "CujsonError", &PyTuple::new(py, [exc])?)?;
    module.add("CujsonError", &cujson_error)?;
    CUJSON_ERROR.set(py, cujson_error.clone().unbind()).ok();

    let invalid_utf8 = make_type(
        py,
        "InvalidUtf8Error",
        &PyTuple::new(py, [cujson_error.clone(), value_error.clone()])?,
    )?;
    module.add("InvalidUtf8Error", &invalid_utf8)?;
    INVALID_UTF8_ERROR.set(py, invalid_utf8.unbind()).ok();

    let unbalanced = make_type(
        py,
        "UnbalancedError",
        &PyTuple::new(py, [cujson_error.clone(), value_error.clone()])?,
    )?;
    module.add("UnbalancedError", &unbalanced)?;
    UNBALANCED_ERROR.set(py, unbalanced.unbind()).ok();

    let input_error = make_type(
        py,
        "InputError",
        &PyTuple::new(py, [cujson_error.clone(), value_error.clone()])?,
    )?;
    module.add("InputError", &input_error)?;
    INPUT_ERROR.set(py, input_error.unbind()).ok();

    let cuda_error = make_type(
        py,
        "CudaError",
        &PyTuple::new(py, [cujson_error.clone(), runtime_error])?,
    )?;
    module.add("CudaError", &cuda_error)?;
    CUDA_ERROR.set(py, cuda_error.clone().unbind()).ok();

    let no_device = make_type(py, "NoDeviceError", &PyTuple::new(py, [cuda_error])?)?;
    module.add("NoDeviceError", &no_device)?;
    NO_DEVICE_ERROR.set(py, no_device.unbind()).ok();

    Ok(())
}

fn err_from(py: Python<'_>, cell: &PyOnceLock<Py<PyType>>, message: String) -> PyErr {
    let ty = cell
        .get(py)
        .expect("exceptions::init must run before any error conversion")
        .bind(py);
    PyErr::from_type(ty.clone(), message)
}

/// Convert a `cujson::Error` into the matching Python exception. Must only
/// be called after [`init`] has run (true for every call reachable from
/// Python, since `init` runs in the `#[pymodule]` function).
pub fn to_pyerr(py: Python<'_>, err: cujson::Error) -> PyErr {
    use cujson::Error as E;
    match err {
        E::NoDevice => err_from(
            py,
            &NO_DEVICE_ERROR,
            "no CUDA device visible (driver responded, device count is 0)".to_string(),
        ),
        E::InvalidUtf8 => err_from(
            py,
            &INVALID_UTF8_ERROR,
            "input is not valid UTF-8".to_string(),
        ),
        E::Unbalanced => err_from(
            py,
            &UNBALANCED_ERROR,
            "unbalanced JSON structure".to_string(),
        ),
        E::InputTooLarge { len, max } => err_from(
            py,
            &INPUT_ERROR,
            format!("input is {len} bytes, cuJSON's limit is {max} bytes"),
        ),
        E::EmptyInput => err_from(py, &INPUT_ERROR, "empty input".to_string()),
        E::NotSingleValue => err_from(py, &INPUT_ERROR, err.to_string()),
        E::Cuda { code, message } => {
            err_from(py, &CUDA_ERROR, format!("CUDA error {code}: {message}"))
        }
        E::Internal => err_from(py, &CUJSON_ERROR, "internal error".to_string()),
        E::Io(e) => err_from(py, &CUJSON_ERROR, format!("I/O error: {e}")),
        other => err_from(py, &CUJSON_ERROR, format!("{other}")),
    }
}

/// Convert a `cujson::tape::Error` (navigator/scalar-decoding errors, e.g.
/// a malformed `\uXXXX` escape or non-UTF-8 string content found while
/// materializing a value) into the matching Python exception.
pub fn to_pyerr_tape(py: Python<'_>, err: cujson::tape::Error) -> PyErr {
    use cujson::tape::Error as E;
    match err {
        E::InvalidUtf8 => err_from(
            py,
            &INVALID_UTF8_ERROR,
            "input is not valid UTF-8".to_string(),
        ),
        E::UnbalancedBrackets => err_from(
            py,
            &UNBALANCED_ERROR,
            "unbalanced JSON structure".to_string(),
        ),
        E::NotSingleValue => err_from(py, &INPUT_ERROR, err.to_string()),
        other => err_from(py, &CUJSON_ERROR, format!("{other}")),
    }
}
