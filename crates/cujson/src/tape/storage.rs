//! Tape array storage.

use std::ops::Deref;

/// Owner of one pinned host allocation returned by the GPU parser
/// (`cujson_sys::cujson_tape`). `structural` and `pair_pos` are two
/// pointers into the *same* allocation (fact established in review of
/// `parse_standard_json.cu:1707-1708`/`parse_json_lines.cu:1290-1291`:
/// `pair_pos = base + n + 1`) — this type owns the whole allocation and is
/// shared (via `Arc`) between the two `TapeStorage::Pinned` slices built
/// from it, so `cujson_tape_free` runs exactly once, on whichever `Arc`
/// clone is dropped last.
#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct PinnedAlloc(cujson_sys::cujson_tape);

// Safety: the wrapped `cujson_tape` is only ever read through the
// `TapeStorage::Pinned` slices built from it (never mutated after
// `tape_from_ffi` constructs it), and `cujson_tape_free` (called from
// `Drop` below) is safe to call from any thread once the allocation is no
// longer referenced.
#[cfg(feature = "cuda")]
unsafe impl Send for PinnedAlloc {}
#[cfg(feature = "cuda")]
unsafe impl Sync for PinnedAlloc {}

#[cfg(feature = "cuda")]
impl Drop for PinnedAlloc {
    fn drop(&mut self) {
        unsafe { cujson_sys::cujson_tape_free(&mut self.0) }
    }
}

/// Backing storage for one tape array (`structural` or `pair_pos`).
#[derive(Debug, Clone)]
pub enum TapeStorage {
    /// Plain heap-allocated array: used by the CPU reference builder and by
    /// hand-written test tapes.
    Owned(Box<[i32]>),
    /// A read-only slice into a pinned host allocation owned by the GPU
    /// parser (task 06). Never exposed as `&mut` — the allocation is freed
    /// exactly once, by `PinnedAlloc::drop`, when the last `Arc` clone
    /// shared between a tape's `structural` and `pair_pos` storage is
    /// dropped.
    #[cfg(feature = "cuda")]
    Pinned {
        alloc: std::sync::Arc<PinnedAlloc>,
        ptr: *const i32,
        len: usize,
    },
}

// Safety: `TapeStorage` never hands out `&mut` access to the pointee of its
// `Pinned` variant's raw pointer, and that pointer stays valid for as long
// as any clone of `alloc` (itself `Send + Sync`, see above) is alive. So a
// `TapeStorage` may be sent to / shared with another thread exactly as
// freely as the `Arc<PinnedAlloc>` and slice it borrows from could be.
unsafe impl Send for TapeStorage {}
unsafe impl Sync for TapeStorage {}

impl Deref for TapeStorage {
    type Target = [i32];

    fn deref(&self) -> &[i32] {
        match self {
            TapeStorage::Owned(b) => b,
            #[cfg(feature = "cuda")]
            TapeStorage::Pinned { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            },
        }
    }
}

impl From<Vec<i32>> for TapeStorage {
    fn from(v: Vec<i32>) -> Self {
        TapeStorage::Owned(v.into_boxed_slice())
    }
}

impl From<Box<[i32]>> for TapeStorage {
    fn from(b: Box<[i32]>) -> Self {
        TapeStorage::Owned(b)
    }
}

/// The two tape arrays, as defined by `tape/FORMAT.md`.
#[derive(Debug, Clone)]
pub struct Tape {
    pub structural: TapeStorage,
    pub pair_pos: TapeStorage,
}

impl Tape {
    /// Total tape length: number of structural entries, including the two
    /// artificial wrapper entries at index `0` and `len() - 1`. See
    /// `tape/FORMAT.md` §2.
    pub fn len(&self) -> usize {
        self.structural.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Wrap a raw `cujson_tape` returned by `cujson_parse_standard`/
/// `cujson_parse_lines` into a `Tape` whose two `TapeStorage::Pinned`
/// arrays share ownership of the one pinned allocation (see `PinnedAlloc`).
#[cfg(feature = "cuda")]
pub(crate) fn tape_from_ffi(raw: cujson_sys::cujson_tape) -> Tape {
    let len = raw.len;
    let structural_ptr = raw.structural as *const i32;
    let pair_pos_ptr = raw.pair_pos as *const i32;
    let alloc = std::sync::Arc::new(PinnedAlloc(raw));
    let structural = TapeStorage::Pinned {
        alloc: alloc.clone(),
        ptr: structural_ptr,
        len,
    };
    let pair_pos = TapeStorage::Pinned {
        alloc,
        ptr: pair_pos_ptr,
        len,
    };
    Tape {
        structural,
        pair_pos,
    }
}
