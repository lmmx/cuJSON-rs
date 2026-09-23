//! Tape array storage.

use std::ops::Deref;

/// Backing storage for one tape array (`structural` or `pair_pos`).
///
/// Kept behind this enum (rather than a bare `Box<[i32]>`) so a later task
/// (06, safe `cujson` crate) can add a variant wrapping a pinned host
/// allocation returned by the GPU parser and free it via `cujson_tape_free`
/// in that variant's `Drop` impl, without any navigator code (which only
/// ever sees `&[i32]` through `Deref`) needing to change.
#[derive(Debug, Clone)]
pub enum TapeStorage {
    /// Plain heap-allocated array: used by the CPU reference builder and by
    /// hand-written test tapes.
    Owned(Box<[i32]>),
}

impl Deref for TapeStorage {
    type Target = [i32];

    fn deref(&self) -> &[i32] {
        match self {
            TapeStorage::Owned(b) => b,
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

/// The two tape arrays plus max nesting depth, as defined by `tape/FORMAT.md`.
#[derive(Debug, Clone)]
pub struct Tape {
    pub structural: TapeStorage,
    pub pair_pos: TapeStorage,
    /// Max bracket-nesting depth, computed by this crate for internal/navigator
    /// use only. **Has no GPU counterpart**: upstream never writes
    /// `cuJSONResult::depth` (`tape/FORMAT.md` §4) — task 10's differential test
    /// must exclude this field from comparison entirely, not mask it.
    pub depth: i32,
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
