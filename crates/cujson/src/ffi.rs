//! Safe wrappers over `cujson-sys`'s `unsafe extern "C"` functions.
//!
//! cuJSON uses the default CUDA stream and keeps no per-call state
//! (`docs/plan/README.md`'s "Concurrency" row), so every entry point here
//! takes one of a process-wide set of slots (one by default) — never call the raw
//! `cujson_sys` functions directly from elsewhere in this crate.

use std::ffi::CStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};

use cujson_sys as sys;

use crate::error::Error;
use crate::tape::{Tape, tape_from_ffi};
use crate::{CudaInfo, DeviceInfo};

/// Parses admitted at once. The default of 1 serialises every FFI call; a
/// higher limit lets parses from different threads overlap on the GPU, each
/// on its own per-thread default stream.
static MAX_CONCURRENT: AtomicUsize = AtomicUsize::new(1);
static IN_FLIGHT: Mutex<usize> = Mutex::new(0);
static SLOT_FREED: Condvar = Condvar::new();

pub(crate) fn set_max_concurrent(n: usize) {
    MAX_CONCURRENT.store(n.max(1), Ordering::Relaxed);
    // `n` tapes being built, one queued and one being read by a consumer.
    unsafe { sys::cujson_pinned_cache_set_limit(n.max(1) + 2) };
    SLOT_FREED.notify_all();
}

/// Holds one of the `MAX_CONCURRENT` slots until dropped.
pub(crate) struct GpuGuard;

impl Drop for GpuGuard {
    fn drop(&mut self) {
        let mut n = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
        *n -= 1;
        SLOT_FREED.notify_one();
    }
}

fn lock() -> GpuGuard {
    let mut n = IN_FLIGHT.lock().unwrap_or_else(|e| e.into_inner());
    while *n >= MAX_CONCURRENT.load(Ordering::Relaxed) {
        n = SLOT_FREED.wait(n).unwrap_or_else(|e| e.into_inner());
    }
    *n += 1;
    GpuGuard
}

/// `cudaGetErrorName`/`cudaGetErrorString` for a raw `cudaError_t`, e.g.
/// `"cudaErrorInsufficientDriver: CUDA driver version is insufficient for
/// CUDA runtime version"` — richer than `format!("cudaError {code}")`, and
/// what the CLI's `info`/`verify` hints (task 07) are built from.
fn cuda_error_string(code: i32) -> String {
    unsafe {
        let ptr = sys::cujson_cuda_error_string(code);
        if ptr.is_null() {
            format!("cudaError {code}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

fn status_message(status: sys::cujson_status) -> String {
    unsafe {
        let ptr = sys::cujson_status_str(status);
        if ptr.is_null() {
            format!("cujson_status {status}")
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    }
}

fn map_error(status: sys::cujson_status, out: &sys::cujson_tape) -> Error {
    match status {
        sys::CUJSON_ERR_UTF8 => Error::InvalidUtf8,
        sys::CUJSON_ERR_UNBALANCED => Error::Unbalanced,
        sys::CUJSON_ERR_EMPTY_INPUT => Error::EmptyInput,
        // Rust's own size check (crate::check_size) runs before every FFI
        // call, so this status should be unreachable in practice; mapped
        // defensively rather than treated as Internal so a future change
        // to that check doesn't silently misreport it.
        sys::CUJSON_ERR_INPUT_TOO_LARGE => Error::InputTooLarge {
            len: 0,
            max: crate::MAX_INPUT_LEN,
        },
        sys::CUJSON_ERR_CUDA => Error::Cuda {
            code: out.cuda_error,
            message: cuda_error_string(out.cuda_error),
        },
        _ => {
            let _ = status_message(status); // best-effort, ignored if unrecognized
            Error::Internal
        }
    }
}

/// Cached result of `cujson_device_count()`: `Ok(count)` (always `>= 1` —
/// `count == 0` is stored as `Err(0)` so the cache and `Error` agree) or
/// `Err(negated cudaError)` from the driver call itself failing. Checked
/// once per process, under `GPU_LOCK`, before the first
/// `parse_standard`/`parse_lines` call: without this, cuJSON's own C++
/// (`capi_standard.cu`/`capi_lines.cu`) can throw a `thrust::system_error`
/// which the shim maps to `CUJSON_ERR_CUDA` — but on some code paths a
/// generic `catch (...)` still lost that to `CUJSON_ERR_INTERNAL`, so this
/// check makes `parse`/`parse_lines` fail the same explicit way
/// `cuda_info()` does (`Error::NoDevice`/`Error::Cuda`) rather than
/// relying on the C++ exception path alone.
static DEVICE_CHECK: OnceLock<Result<i32, i32>> = OnceLock::new();

/// Must be called with `GPU_LOCK` held.
fn device_available() -> Result<(), Error> {
    let result = *DEVICE_CHECK.get_or_init(|| {
        let count = unsafe { sys::cujson_device_count() };
        if count < 0 { Err(-count) } else { Ok(count) }
    });
    match result {
        Ok(count) if count > 0 => Ok(()),
        Ok(_) => Err(Error::NoDevice),
        Err(code) => Err(Error::Cuda {
            code,
            message: cuda_error_string(code),
        }),
    }
}

pub(crate) fn parse_standard(data: &[u8]) -> Result<Tape, Error> {
    let _guard = lock();
    device_available()?;
    let mut out = sys::cujson_tape::default();
    let status = unsafe { sys::cujson_parse_standard(data.as_ptr(), data.len(), &mut out) };
    if status != sys::CUJSON_OK {
        return Err(map_error(status, &out));
    }
    Ok(tape_from_ffi(out))
}

pub(crate) fn parse_lines(data: &[u8], chunk_bytes: usize) -> Result<Tape, Error> {
    let _guard = lock();
    device_available()?;
    let mut out = sys::cujson_tape::default();
    let status =
        unsafe { sys::cujson_parse_lines(data.as_ptr(), data.len(), chunk_bytes, &mut out) };
    if status != sys::CUJSON_OK {
        return Err(map_error(status, &out));
    }
    Ok(tape_from_ffi(out))
}

pub(crate) fn cuda_info() -> Result<CudaInfo, Error> {
    let _guard = lock();

    let version = unsafe { sys::cujson_cuda_runtime_version() };
    if version < 0 {
        return Err(Error::Cuda {
            code: -version,
            message: cuda_error_string(-version),
        });
    }

    let driver_version = unsafe { sys::cujson_cuda_driver_version() };

    // On a machine with no NVIDIA driver at all this is the graceful path:
    // cudaGetDeviceCount returns an error code (typically
    // cudaErrorInsufficientDriver/cudaErrorNoDevice) rather than crashing,
    // so this becomes an `Err`, never a panic.
    device_available()?;
    let count = unsafe { sys::cujson_device_count() };

    let mut devices = Vec::with_capacity(count as usize);
    for index in 0..count {
        let mut buf = [0 as core::ffi::c_char; 256];
        let status = unsafe { sys::cujson_device_name(index, buf.as_mut_ptr(), buf.len()) };
        let name = if status == sys::CUJSON_OK {
            unsafe { CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned() }
        } else {
            format!("<unknown: {}>", status_message(status))
        };
        devices.push(DeviceInfo { index, name });
    }

    let compiled_archs = unsafe {
        let ptr = sys::cujson_compiled_archs();
        if ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    Ok(CudaInfo {
        runtime_version: version,
        devices,
        compiled_archs,
        driver_version,
    })
}

/// No `device_available()` gate — safe to call after a failed
/// `cuda_info()`/`parse*` to help build a hint message (see
/// `crate::driver_version`'s doc comment).
pub(crate) fn driver_version() -> i32 {
    let _guard = lock();
    unsafe { sys::cujson_cuda_driver_version() }
}

pub(crate) fn runtime_version() -> Result<i32, Error> {
    let _guard = lock();
    let version = unsafe { sys::cujson_cuda_runtime_version() };
    if version < 0 {
        return Err(Error::Cuda {
            code: -version,
            message: cuda_error_string(-version),
        });
    }
    Ok(version)
}

pub(crate) fn mem_get_info() -> Result<(usize, usize), Error> {
    let _guard = lock();
    device_available()?;
    let mut free = 0usize;
    let mut total = 0usize;
    let status = unsafe { sys::cujson_mem_get_info(&mut free, &mut total) };
    if status != sys::CUJSON_OK {
        return Err(Error::Internal);
    }
    Ok((free, total))
}
