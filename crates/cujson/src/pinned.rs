//! Page-locked host buffers for parser input.

use std::ops::{Deref, DerefMut};

use crate::Error;

/// A byte buffer in page-locked (pinned) host memory when built with the
/// `cuda` feature, and in ordinary memory otherwise.
///
/// The parsers copy their input to the GPU; from pinned memory that copy runs
/// at close to the PCIe rate instead of staging through a bounce buffer, so
/// decoding input straight into a `PinnedBuffer` and passing it to
/// [`parse_lines`](crate::parse_lines) is faster than passing a `Vec`.
/// Allocating and freeing pinned memory is slow (milliseconds per tens of
/// MB), so keep and reuse buffers rather than making one per parse.
pub struct PinnedBuffer {
    #[cfg(feature = "cuda")]
    ptr: *mut u8,
    #[cfg(not(feature = "cuda"))]
    data: Vec<u8>,
    len: usize,
}

// SAFETY: the buffer is uniquely owned; the raw pointer is never aliased
// and is freed once, on drop.
#[cfg(feature = "cuda")]
unsafe impl Send for PinnedBuffer {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// A zero-filled buffer of `len` bytes.
    pub fn new(len: usize) -> Result<Self, Error> {
        #[cfg(feature = "cuda")]
        {
            if len == 0 {
                return Ok(PinnedBuffer {
                    ptr: std::ptr::NonNull::<u8>::dangling().as_ptr(),
                    len: 0,
                });
            }
            // SAFETY: plain allocation call; a null return is handled below.
            let ptr = unsafe { cujson_sys::cujson_host_alloc(len) }.cast::<u8>();
            if ptr.is_null() {
                return Err(Error::Cuda {
                    code: 2,
                    message: format!("cudaMallocHost of {len} bytes failed"),
                });
            }
            // SAFETY: `ptr` is a fresh allocation of `len` bytes.
            unsafe { ptr.write_bytes(0, len) };
            Ok(PinnedBuffer { ptr, len })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(PinnedBuffer {
                data: vec![0; len],
                len,
            })
        }
    }

    /// A buffer holding a copy of `bytes`.
    pub fn from_slice(bytes: &[u8]) -> Result<Self, Error> {
        let mut buf = Self::new(bytes.len())?;
        buf.copy_from_slice(bytes);
        Ok(buf)
    }
}

impl Deref for PinnedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        #[cfg(feature = "cuda")]
        // SAFETY: `ptr` points to `len` initialised bytes owned by `self`.
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.len)
        }
        #[cfg(not(feature = "cuda"))]
        &self.data[..self.len]
    }
}

impl DerefMut for PinnedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        #[cfg(feature = "cuda")]
        // SAFETY: as in `deref`, and `&mut self` gives unique access.
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.len)
        }
        #[cfg(not(feature = "cuda"))]
        &mut self.data[..self.len]
    }
}

#[cfg(feature = "cuda")]
impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        if self.len != 0 {
            // SAFETY: allocated by `cujson_host_alloc`, freed exactly once.
            unsafe { cujson_sys::cujson_host_free(self.ptr.cast()) }
        }
    }
}

impl std::fmt::Debug for PinnedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .finish()
    }
}
