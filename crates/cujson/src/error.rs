//! Top-level errors for `parse`/`parse_lines`/`parse_file`/`cuda_info`.
//!
//! Distinct from `tape::Error` (tape-building/navigation errors used by the
//! `cpu-reference` path and by `Document`/`Node` accessors).

use std::fmt;
use std::io;

/// Errors from the GPU-parsing entry points (`crate::parse` and friends)
/// and `crate::cuda_info`.
#[non_exhaustive]
#[derive(Debug)]
pub enum Error {
    /// Built without the `cuda` feature — no FFI declarations exist to call.
    CudaNotCompiled,
    /// `cujson_device_count()` returned `0`: a CUDA runtime is linked and a
    /// driver responded, but no GPU is visible.
    NoDevice,
    /// `cujson_status` was `CUJSON_ERR_UTF8`.
    InvalidUtf8,
    /// `cujson_status` was `CUJSON_ERR_UNBALANCED`.
    Unbalanced,
    /// Rejected before reaching the FFI boundary: `len` is within
    /// `-8` of `i32::MAX` (cuJSON uses `int` sizes; see
    /// `capi_standard.cu`'s `INT32_MAX - 8` check, which this mirrors on
    /// the Rust side so an oversized `Vec`/slice never gets to C).
    InputTooLarge {
        len: usize,
        max: usize,
    },
    /// Input was empty (also returned by `cujson_status` `CUJSON_ERR_EMPTY_INPUT`).
    EmptyInput,
    /// A CUDA runtime call failed: `code` is a raw `cudaError_t` value,
    /// `message` a short human-readable description built on the Rust
    /// side (the C ABI has no `cudaGetErrorString` wrapper).
    Cuda {
        code: i32,
        message: String,
    },
    /// `cujson_status` was `CUJSON_ERR_INTERNAL`, or another status value
    /// this crate doesn't otherwise map (defensive: keeps a future header
    /// addition from being undefined behavior to receive).
    Internal,
    Io(io::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::CudaNotCompiled => write!(f, "cujson was built without the `cuda` feature"),
            Error::NoDevice => write!(
                f,
                "no CUDA device visible (driver responded, device count is 0)"
            ),
            Error::InvalidUtf8 => write!(f, "input is not valid UTF-8"),
            Error::Unbalanced => write!(f, "unbalanced JSON structure"),
            Error::InputTooLarge { len, max } => {
                write!(f, "input is {len} bytes, cuJSON's limit is {max} bytes")
            }
            Error::EmptyInput => write!(f, "empty input"),
            Error::Cuda { code, message } => write!(f, "CUDA error {code}: {message}"),
            Error::Internal => write!(f, "internal error"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}
