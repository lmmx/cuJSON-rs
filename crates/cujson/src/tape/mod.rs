//! Tape format, CPU reference builder, and navigator.
//!
//! See `FORMAT.md` in this directory for the tape layout this module
//! implements, cited to the upstream CUDA/C++ source.

mod document;
mod error;
mod storage;

#[cfg(feature = "cpu-reference")]
mod builder;
#[cfg(feature = "cpu-reference")]
mod diff;

pub use document::{Document, Kind, Node};
pub use error::Error;
pub(crate) use error::NOT_SINGLE_VALUE;
pub use storage::{Tape, TapeStorage};

#[cfg(feature = "cuda")]
pub(crate) use storage::tape_from_ffi;

#[cfg(feature = "cpu-reference")]
pub use builder::{Mode, build_tape_cpu};
#[cfg(feature = "cpu-reference")]
pub use diff::{DiffField, TapeDiff, diff_tapes};
