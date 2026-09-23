//! Tape format, CPU reference builder, and navigator.
//!
//! See `FORMAT.md` in this directory for the tape layout this module
//! implements, cited to the upstream CUDA/C++ source.

mod document;
mod error;
mod storage;

#[cfg(feature = "cpu-reference")]
mod builder;

pub use document::{Document, Kind, Node};
pub use error::Error;
pub use storage::{Tape, TapeStorage};

#[cfg(feature = "cpu-reference")]
pub use builder::{Mode, build_tape_cpu};
