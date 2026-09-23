//! Direct `cujson::tape::Node` -> `Py<PyAny>` conversion, walking the tape
//! once per node without an intermediate `serde_json::Value` — the
//! "preferred, direct" path task 08's brief calls out.

use cujson::tape::{Kind, Node};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::exceptions;

pub fn node_to_object(py: Python<'_>, node: &Node<'_>) -> PyResult<Py<PyAny>> {
    match node.kind() {
        Kind::Null => Ok(py.None()),
        Kind::Bool => Ok(node
            .as_bool()
            .map_err(|e| exceptions::to_pyerr_tape(py, e))?
            .into_pyobject(py)?
            .to_owned()
            .into_any()
            .unbind()),
        Kind::Number => number_to_object(py, node),
        Kind::String => {
            let s = node
                .as_str()
                .map_err(|e| exceptions::to_pyerr_tape(py, e))?;
            Ok(s.into_pyobject(py)?.into_any().unbind())
        }
        Kind::Array => {
            let list = PyList::empty(py);
            for child in node.iter_array().expect("Kind::Array has iter_array") {
                list.append(node_to_object(py, &child)?)?;
            }
            Ok(list.into_any().unbind())
        }
        Kind::Object => {
            let dict = PyDict::new(py);
            for (key, value) in node.iter_object().expect("Kind::Object has iter_object") {
                dict.set_item(key.as_ref(), node_to_object(py, &value)?)?;
            }
            Ok(dict.into_any().unbind())
        }
    }
}

/// Numbers without a `.`/`e`/`E` parse as `i64` (falling back to `u64` for
/// values above `i64::MAX`), matching `Node::to_value`'s int-vs-float
/// split (`crates/cujson/src/tape/document.rs`) so both conversion paths
/// agree on which numbers come back as Python `int` vs `float`.
fn number_to_object(py: Python<'_>, node: &Node<'_>) -> PyResult<Py<PyAny>> {
    let raw = std::str::from_utf8(node.raw()).unwrap_or("0");
    if !raw.contains(['.', 'e', 'E']) {
        if let Ok(i) = raw.parse::<i64>() {
            return Ok(i.into_pyobject(py)?.into_any().unbind());
        }
        if let Ok(u) = raw.parse::<u64>() {
            return Ok(u.into_pyobject(py)?.into_any().unbind());
        }
    }
    let f: f64 = raw
        .parse()
        .map_err(|_| exceptions::to_pyerr_tape(py, cujson::tape::Error::InvalidNumber))?;
    Ok(f.into_pyobject(py)?.into_any().unbind())
}
