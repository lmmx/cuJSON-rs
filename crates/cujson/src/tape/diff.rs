//! GPU-vs-CPU tape comparison (`tape/FORMAT.md` §3 and Unresolved).
//!
//! One reusable function so both the GPU integration tests
//! (`crates/cujson/tests/gpu.rs`) and (in a later task) the `cujson verify`
//! CLI command compare tapes the same way, with the same undefined-slot
//! exclusions.

use std::fmt;

use super::storage::Tape;

/// Which array a [`TapeDiff`] was found in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiffField {
    /// `Tape::structural` — compared at every index.
    Structural,
    /// `Tape::pair_pos` — compared only at opener (`{`/`[`) indices among
    /// real entries `1..len-2`; see `tape/FORMAT.md` §3.
    PairPos,
    /// The two tapes have different `Tape::len()`.
    Length,
}

/// The first mismatch found by [`diff_tapes`], with enough context to print
/// directly (e.g. by `cujson verify`).
#[derive(Debug, Clone)]
pub struct TapeDiff {
    pub index: usize,
    pub field: DiffField,
    pub a: i32,
    pub b: i32,
    /// A short window of the original input around the mismatch's byte
    /// offset, or empty when no byte offset applies (e.g. a length
    /// mismatch).
    pub context: String,
}

impl fmt::Display for TapeDiff {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "tape mismatch at index {} ({:?}): a={} b={}",
            self.index, self.field, self.a, self.b
        )?;
        if !self.context.is_empty() {
            write!(f, " — near input: {:?}", self.context)?;
        }
        Ok(())
    }
}

/// Mirrors the private `Document::get_char`/`byte_pos` logic
/// (`tape/document.rs`) just enough to classify a tape index as an opener,
/// without needing a full `Document`. `t` supplies the tape whose
/// `structural` entry at `idx` is read; `idx` must be a "real" entry
/// (`1..t.len()-1`) — callers only call this inside that range.
fn tape_char(input: &[u8], t: &Tape, idx: usize) -> u8 {
    let pos = t.structural[idx] - 1;
    if pos < 0 || pos as usize >= input.len() {
        return 0;
    }
    let c = input[pos as usize];
    if c == b'\n' { b',' } else { c }
}

fn context_near(input: &[u8], byte_pos: Option<usize>) -> String {
    match byte_pos {
        Some(p) if p < input.len() => {
            let start = p.saturating_sub(16);
            let end = (p + 16).min(input.len());
            String::from_utf8_lossy(&input[start..end]).into_owned()
        }
        _ => String::new(),
    }
}

/// Compare a GPU-produced tape (`a`) against a CPU-reference tape (`b`) —
/// argument order doesn't matter for the comparison, only for which value
/// ends up in `TapeDiff::a`/`b` — for the same `input`, returning the first
/// mismatch found or `None` if they agree everywhere the format defines an
/// agreement:
///
/// - `structural` is compared at every index.
/// - `pair_pos` is compared **only** at indices whose tape character (per
///   `a`, since a structural mismatch would already have been reported) is
///   an opening bracket (`{` or `[`), among real entries `1..len-2`. Every
///   other `pair_pos` slot is undefined on the GPU side (`tape/FORMAT.md`
///   §3, Unresolved #2/#3) and is never compared.
/// - `Tape::depth()` is never compared — it has no GPU counterpart
///   (`tape/FORMAT.md` §4).
pub fn diff_tapes(input: &[u8], a: &Tape, b: &Tape) -> Option<TapeDiff> {
    if a.len() != b.len() {
        return Some(TapeDiff {
            index: 0,
            field: DiffField::Length,
            a: a.len() as i32,
            b: b.len() as i32,
            context: String::new(),
        });
    }
    let n = a.len();
    for i in 0..n {
        if a.structural[i] != b.structural[i] {
            let byte_pos = if i == 0 || i + 1 == n {
                None
            } else {
                Some((a.structural[i] - 1).max(0) as usize)
            };
            return Some(TapeDiff {
                index: i,
                field: DiffField::Structural,
                a: a.structural[i],
                b: b.structural[i],
                context: context_near(input, byte_pos),
            });
        }
    }
    if n < 2 {
        return None;
    }
    for i in 1..n - 1 {
        let c = tape_char(input, a, i);
        if c != b'{' && c != b'[' {
            continue;
        }
        if a.pair_pos[i] != b.pair_pos[i] {
            let byte_pos = Some((a.structural[i] - 1).max(0) as usize);
            return Some(TapeDiff {
                index: i,
                field: DiffField::PairPos,
                a: a.pair_pos[i],
                b: b.pair_pos[i],
                context: context_near(input, byte_pos),
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::{Mode, build_tape_cpu};

    #[test]
    fn identical_tapes_have_no_diff() {
        let input = b"{\"a\":[1,2],\"b\":3}";
        let a = build_tape_cpu(input, Mode::Standard).unwrap();
        let b = build_tape_cpu(input, Mode::Standard).unwrap();
        assert!(diff_tapes(input, &a, &b).is_none());
    }

    fn with_slot(
        storage: &crate::tape::TapeStorage,
        idx: usize,
        f: impl FnOnce(&mut i32),
    ) -> crate::tape::TapeStorage {
        let mut v = storage.to_vec();
        f(&mut v[idx]);
        v.into()
    }

    #[test]
    fn structural_mismatch_detected() {
        let input = b"{\"a\":1}";
        let a = build_tape_cpu(input, Mode::Standard).unwrap();
        let mut b = build_tape_cpu(input, Mode::Standard).unwrap();
        // Corrupt one structural entry.
        b.structural = with_slot(&b.structural, 1, |v| *v += 1);
        let diff = diff_tapes(input, &a, &b).unwrap();
        assert_eq!(diff.field, DiffField::Structural);
        assert_eq!(diff.index, 1);
    }

    #[test]
    fn non_opener_pair_pos_mismatch_is_ignored() {
        let input = b"{\"a\":1}";
        let a = build_tape_cpu(input, Mode::Standard).unwrap();
        let mut b = build_tape_cpu(input, Mode::Standard).unwrap();
        // Corrupt a non-opener pair_pos slot (index 2 is ':'); must not be
        // reported, matching the undefined-slot exclusion.
        b.pair_pos = with_slot(&b.pair_pos, 2, |v| *v = 999);
        assert!(diff_tapes(input, &a, &b).is_none());
    }

    #[test]
    fn opener_pair_pos_mismatch_detected() {
        let input = b"{\"a\":1}";
        let a = build_tape_cpu(input, Mode::Standard).unwrap();
        let mut b = build_tape_cpu(input, Mode::Standard).unwrap();
        // Index 1 is `{`, an opener.
        b.pair_pos = with_slot(&b.pair_pos, 1, |v| *v += 1);
        let diff = diff_tapes(input, &a, &b).unwrap();
        assert_eq!(diff.field, DiffField::PairPos);
        assert_eq!(diff.index, 1);
    }
}
