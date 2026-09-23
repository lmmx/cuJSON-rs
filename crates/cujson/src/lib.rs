//! Safe Rust bindings to [cuJSON](https://github.com/) — a GPU JSON parser
//! from the paper *cuJSON: A GPU-Accelerated JSON Parser* (ASPLOS '26,
//! cited in the workspace `Cargo.toml` description). This crate calls the
//! upstream CUDA kernels through the plain C ABI in
//! `crates/cujson-sys/cuda/cujson_capi.h` and exposes the resulting tape
//! (`tape::Tape`) through a `Document`/`Node` navigator
//! (`crates/cujson/src/tape/document.rs`).
//!
//! # The `cuda` feature
//!
//! Without it (the default), every parsing function returns
//! `Err(Error::CudaNotCompiled)` — the crate still builds and its
//! navigator/CPU-reference code (`cpu-reference` feature) still works, but
//! no FFI declarations exist to call. With it, `build.rs`
//! (`crates/cujson-sys/build.rs`) compiles upstream's CUDA sources with
//! `nvcc` and links `cudart_static`, so the resulting binary depends only
//! on the NVIDIA driver (`libcuda.so.1`, loaded at runtime) — no CUDA
//! shared libraries need to be installed or bundled. `CUJSON_CUDA_ARCHS`
//! (an env var read by that build script) overrides the default compiled
//! architecture list, e.g. `CUJSON_CUDA_ARCHS=80` for a fast local build
//! targeting one GPU generation.
//!
//! # Limits
//!
//! cuJSON uses `int`-sized token counts internally. Inputs within 8 bytes
//! of `i32::MAX` (see [`MAX_INPUT_LEN`]) are rejected by this crate with
//! [`Error::InputTooLarge`] before ever reaching C++.
//!
//! # Example
//!
//! ```no_run
//! # #[cfg(feature = "cuda")]
//! # fn main() -> Result<(), cujson::Error> {
//! let doc = cujson::parse(br#"{"a": [1, 2, 3]}"#)?;
//! let a = doc.root().get("a").unwrap();
//! assert_eq!(a.index(1).unwrap().as_i64(), Ok(2));
//! # Ok(())
//! # }
//! # #[cfg(not(feature = "cuda"))]
//! # fn main() {}
//! ```

pub mod tape;

mod error;
#[cfg(feature = "cuda")]
mod ffi;

pub use error::Error;
pub use tape::Document;

#[cfg(feature = "cuda")]
use std::borrow::Cow;
use std::path::Path;

/// Largest input length this crate will pass to cuJSON. cuJSON pads its
/// input up to a 4-byte boundary and works in `int`-sized token counts;
/// `capi_standard.cu`/`capi_lines.cu` reject `size >= INT32_MAX - 8` at the
/// FFI boundary, and this constant mirrors that check so an oversized
/// input is rejected on the Rust side with a typed error instead of
/// reaching C++.
pub const MAX_INPUT_LEN: usize = i32::MAX as usize - 8;

/// Options for [`parse_lines`].
#[derive(Debug, Clone, Copy)]
pub struct LinesOptions {
    /// Upper bound on each chunk's size in bytes; a single line longer than
    /// this still gets a chunk to itself
    /// (`crates/cujson-sys/cuda/capi_lines.cu`'s `build_lines_chunks`). `0`
    /// is treated by `cujson_parse_lines` as "one chunk covering the whole
    /// input" (`capi_lines.cu:73`).
    pub chunk_bytes: usize,
}

impl Default for LinesOptions {
    fn default() -> Self {
        // Matches upstream's own default (main_jsonlines_chunksize_MB.cu:16,
        // `size_t maxChunkSizeMB = 256;`) in the vendored reference copy at
        // `vendor/cuJSON/main_jsonlines_chunksize_MB.cu`.
        LinesOptions {
            chunk_bytes: 256 * 1024 * 1024,
        }
    }
}

/// CUDA runtime/device info, returned by [`cuda_info`].
#[derive(Debug, Clone)]
pub struct CudaInfo {
    /// e.g. `12080` for CUDA 12.8 (`cudaRuntimeGetVersion`'s encoding).
    pub runtime_version: i32,
    pub devices: Vec<DeviceInfo>,
    /// The build's `-DCUJSON_COMPILED_ARCHS` string, e.g. `"75,80,86,89,90;ptx90"`.
    pub compiled_archs: String,
    /// The driver's supported CUDA version (`cudaDriverGetVersion`'s
    /// encoding, same as `runtime_version`), or `0` if no driver responds.
    pub driver_version: i32,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: i32,
    pub name: String,
}

fn check_size(len: usize) -> Result<(), Error> {
    if len == 0 {
        return Err(Error::EmptyInput);
    }
    if len >= MAX_INPUT_LEN {
        return Err(Error::InputTooLarge {
            len,
            max: MAX_INPUT_LEN,
        });
    }
    Ok(())
}

/// Standard mode accepts exactly one top-level value; cuJSON itself does not
/// check this (see [`Document::is_single_value`]).
#[cfg(feature = "cuda")]
fn single_value(doc: Document<'_>) -> Result<Document<'_>, Error> {
    if doc.is_single_value() {
        Ok(doc)
    } else {
        Err(Error::NotSingleValue)
    }
}

/// Parse `input` as standard JSON on the GPU. Borrows `input` for the
/// lifetime of the returned `Document`.
pub fn parse(input: &[u8]) -> Result<Document<'_>, Error> {
    check_size(input.len())?;
    #[cfg(feature = "cuda")]
    {
        let tape = ffi::parse_standard(input)?;
        single_value(Document::new(Cow::Borrowed(input), tape))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = input;
        Err(Error::CudaNotCompiled)
    }
}

/// Parse owned `input` as standard JSON on the GPU, returning a
/// `'static` `Document` that owns its bytes.
pub fn parse_owned(input: Vec<u8>) -> Result<Document<'static>, Error> {
    check_size(input.len())?;
    #[cfg(feature = "cuda")]
    {
        let tape = ffi::parse_standard(&input)?;
        single_value(Document::new(Cow::Owned(input), tape))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = input;
        Err(Error::CudaNotCompiled)
    }
}

/// Parse `input` as JSON Lines (newline-delimited JSON) on the GPU. See
/// `tape/FORMAT.md` §5 for the tape this produces — every top-level value
/// is reachable via `Document::lines()`.
pub fn parse_lines(input: &[u8], opts: LinesOptions) -> Result<Document<'_>, Error> {
    check_size(input.len())?;
    #[cfg(feature = "cuda")]
    {
        let tape = ffi::parse_lines(input, opts.chunk_bytes)?;
        Ok(Document::new(Cow::Borrowed(input), tape))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (input, opts);
        Err(Error::CudaNotCompiled)
    }
}

/// Parse owned `input` as JSON Lines on the GPU, returning a `'static`
/// `Document` that owns its bytes (mirrors [`parse_owned`] for
/// [`parse_lines`]).
pub fn parse_lines_owned(input: Vec<u8>, opts: LinesOptions) -> Result<Document<'static>, Error> {
    check_size(input.len())?;
    #[cfg(feature = "cuda")]
    {
        let tape = ffi::parse_lines(&input, opts.chunk_bytes)?;
        Ok(Document::new(Cow::Owned(input), tape))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (input, opts);
        Err(Error::CudaNotCompiled)
    }
}

/// Read `path` and parse it as standard JSON on the GPU.
pub fn parse_file(path: impl AsRef<Path>) -> Result<Document<'static>, Error> {
    let bytes = std::fs::read(path)?;
    parse_owned(bytes)
}

/// CUDA runtime version, visible devices, and this build's compiled
/// architecture list. Without the `cuda` feature, always
/// `Err(Error::CudaNotCompiled)`. On a machine with no NVIDIA driver
/// installed, `Err(Error::Cuda { .. })` (the driver call itself failed) or
/// `Err(Error::NoDevice)` (the driver responded but reports zero devices) —
/// never a panic.
pub fn cuda_info() -> Result<CudaInfo, Error> {
    #[cfg(feature = "cuda")]
    {
        ffi::cuda_info()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Error::CudaNotCompiled)
    }
}

/// The driver's supported CUDA version (`cudaDriverGetVersion`'s
/// encoding), or `0` if no driver responds. Unlike [`cuda_info`], this
/// never errors on a driverless/deviceless host — it's meant to be called
/// *after* [`cuda_info`] or a `parse*` call has already failed, so a
/// caller (e.g. `cujson verify`) can distinguish "no driver at all"
/// (`driver_version() == 0`) from "driver present but too old for this
/// binary's CUDA runtime" (`driver_version() > 0` but below
/// `CudaInfo::runtime_version`) when building a hint message.
pub fn driver_version() -> i32 {
    #[cfg(feature = "cuda")]
    {
        ffi::driver_version()
    }
    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// The CUDA runtime version this binary was built against
/// (`cudaRuntimeGetVersion`'s encoding), independent of whether a device
/// or driver is present — pairs with [`driver_version`] to build a hint
/// message when [`cuda_info`] fails.
pub fn runtime_version() -> Result<i32, Error> {
    #[cfg(feature = "cuda")]
    {
        ffi::runtime_version()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Error::CudaNotCompiled)
    }
}

/// Free/total device memory in bytes (`cudaMemGetInfo`, after a
/// `cudaDeviceSynchronize`). Used by `cujson verify`'s memory-growth
/// heuristic (task 07); not called by `parse`/`parse_lines`.
pub fn device_memory() -> Result<(usize, usize), Error> {
    #[cfg(feature = "cuda")]
    {
        ffi::mem_get_info()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(Error::CudaNotCompiled)
    }
}

/// The CPU reference parser (`cpu-reference` feature), kept in its own
/// module and never called by [`parse`]/[`parse_lines`]/[`parse_file`] —
/// so that nobody benchmarks this path while believing it's the GPU one.
/// Its tape is bit-for-bit what `tape/FORMAT.md` documents the GPU as
/// producing, but that equivalence is a hypothesis until confirmed by a
/// real GPU (`tape::diff_tapes`, task 10).
#[cfg(feature = "cpu-reference")]
pub mod cpu {
    use std::borrow::Cow;

    pub use crate::tape::Mode;
    use crate::tape::{Document, Error, build_tape_cpu};

    /// Parse `input` on the CPU, using the same tape format as the GPU
    /// path (`tape/FORMAT.md`).
    pub fn parse(input: &[u8], mode: Mode) -> Result<Document<'_>, Error> {
        let tape = build_tape_cpu(input, mode)?;
        let doc = Document::new(Cow::Borrowed(input), tape);
        if mode == Mode::Standard && !doc.is_single_value() {
            return Err(Error::NotSingleValue);
        }
        Ok(doc)
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests_no_cuda {
    #[test]
    fn parse_without_cuda_feature_errors() {
        let err = super::parse(b"{}").unwrap_err();
        assert!(matches!(err, super::Error::CudaNotCompiled));
    }

    #[test]
    fn cuda_info_without_cuda_feature_errors() {
        let err = super::cuda_info().unwrap_err();
        assert!(matches!(err, super::Error::CudaNotCompiled));
    }

    #[test]
    fn empty_input_is_rejected_before_cuda_check() {
        // Size checking runs first regardless of the `cuda` feature, so
        // this must not report CudaNotCompiled instead.
        let err = super::parse(b"").unwrap_err();
        assert!(matches!(err, super::Error::EmptyInput));
    }

    #[test]
    fn parse_lines_owned_without_cuda_feature_errors() {
        let err =
            super::parse_lines_owned(b"{}".to_vec(), super::LinesOptions::default()).unwrap_err();
        assert!(matches!(err, super::Error::CudaNotCompiled));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests_cuda_no_device {
    //! This container has no NVIDIA driver installed (task 06 brief). These
    //! tests confirm the graceful-error path at tier 2 (compiled with
    //! `cuda`, run with no GPU) rather than a panic or crash — they do not
    //! confirm GPU behavior (tier 3, `crates/cujson/tests/gpu.rs`).

    #[test]
    fn cuda_info_errors_without_a_driver() {
        let result = super::cuda_info();
        assert!(
            result.is_err(),
            "expected an Err (no driver/device here), got {result:?}"
        );
    }

    #[test]
    fn parse_errors_without_a_driver() {
        let result = super::parse(b"{}");
        assert!(
            result.is_err(),
            "expected an Err (no driver/device here), got {}",
            result.is_ok()
        );
    }
}
