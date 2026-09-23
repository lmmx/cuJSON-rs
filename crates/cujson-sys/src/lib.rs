//! Raw FFI bindings to cuJSON.
//!
//! Mirrors `crates/cujson-sys/cuda/cujson_capi.h` field-for-field and
//! function-for-function. `#![no_std]`-compatible: everything here is
//! `#[repr(C)]` primitives and raw pointers, no allocation.
#![no_std]
#![allow(non_camel_case_types)]

#[cfg(feature = "cuda")]
use core::ffi::{c_char, c_int};

/// Mirrors `cujson_status` (cujson_capi.h). A plain `i32`, not a Rust
/// `enum`, because the value comes from C and an out-of-range value from a
/// future header change must not be undefined behavior to hold.
pub type cujson_status = i32;

pub const CUJSON_OK: cujson_status = 0;
pub const CUJSON_ERR_UTF8: cujson_status = 1;
pub const CUJSON_ERR_UNBALANCED: cujson_status = 2;
pub const CUJSON_ERR_INPUT_TOO_LARGE: cujson_status = 3;
pub const CUJSON_ERR_CUDA: cujson_status = 4;
pub const CUJSON_ERR_INTERNAL: cujson_status = 5;
pub const CUJSON_ERR_EMPTY_INPUT: cujson_status = 6;

/// Mirrors `cujson_tape` (cujson_capi.h). Layout verified against a C
/// compile of the header by `tests/layout.rs` (tier 1, no nvcc needed).
#[repr(C)]
#[derive(Debug)]
pub struct cujson_tape {
    pub structural: *mut i32,
    pub pair_pos: *mut i32,
    pub len: usize,
    pub cuda_error: i32,
    pub _alloc: *mut core::ffi::c_void,
}

impl Default for cujson_tape {
    fn default() -> Self {
        cujson_tape {
            structural: core::ptr::null_mut(),
            pair_pos: core::ptr::null_mut(),
            len: 0,
            cuda_error: 0,
            _alloc: core::ptr::null_mut(),
        }
    }
}

#[cfg(feature = "cuda")]
unsafe extern "C" {
    pub fn cujson_parse_standard(
        data: *const u8,
        size: usize,
        out: *mut cujson_tape,
    ) -> cujson_status;
    pub fn cujson_parse_lines(
        data: *const u8,
        size: usize,
        chunk_bytes: usize,
        out: *mut cujson_tape,
    ) -> cujson_status;
    pub fn cujson_tape_free(tape: *mut cujson_tape);
    pub fn cujson_pinned_cache_trim();
    pub fn cujson_pinned_cache_set_limit(buffers: usize);
    pub fn cujson_host_alloc(bytes: usize) -> *mut core::ffi::c_void;
    pub fn cujson_host_free(p: *mut core::ffi::c_void);
    pub fn cujson_status_str(s: cujson_status) -> *const c_char;
    pub fn cujson_cuda_runtime_version() -> c_int;
    pub fn cujson_device_count() -> c_int;
    pub fn cujson_device_name(device: c_int, buf: *mut c_char, buf_len: usize) -> cujson_status;
    pub fn cujson_compiled_archs() -> *const c_char;
    pub fn cujson_cuda_error_string(err: c_int) -> *const c_char;
    pub fn cujson_cuda_driver_version() -> c_int;
    pub fn cujson_mem_get_info(free_bytes: *mut usize, total_bytes: *mut usize) -> cujson_status;
}
