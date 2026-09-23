//! CPU reference tape builder (`cpu-reference` feature).
//!
//! Produces exactly the tape defined in `tape/FORMAT.md`: a straightforward
//! sequential scan, not optimised for speed. It serves as the fixture
//! generator for navigator tests and as the oracle for task 10's
//! differential test against real GPU output (unverified here — no GPU in
//! this container).

use super::error::Error;
use super::storage::{Tape, TapeStorage};

/// Input shape: a single JSON document, or a newline-delimited stream of
/// JSON documents (JSON Lines). See `tape/FORMAT.md` §5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Standard,
    Lines,
}

/// Structural offsets, opener/closer index pairs, and max depth for a
/// (possibly multi-line) document — the pre-tape intermediate shared by
/// `Mode::Standard` and `Mode::Lines`.
struct RawStructure {
    offsets: Vec<i32>,
    pairs: Vec<(usize, usize)>,
    depth: i32,
}

/// Scan `bytes` for structural characters (`tape/FORMAT.md` §1), returning
/// their 1-based offsets into the document that starts at `base` (0-based),
/// the (opener, closer) index pairs into that offsets list, and the max
/// bracket-nesting depth.
///
/// `mark_newline` mirrors the one difference between the Standard and Lines
/// `bitMapCreatorSimd` variants (`tape/FORMAT.md` §1/§5): when set, an
/// unescaped `\n` outside a string is folded into the *same* structural
/// bitmap as `{}[]:,` (masked by the same in-string exclusion), not treated
/// specially.
fn scan_structural(bytes: &[u8], base: usize, mark_newline: bool) -> Result<RawStructure, Error> {
    let mut offsets = Vec::new();
    let mut pairs = Vec::new();
    let mut stack: Vec<(u8, usize)> = Vec::new();
    let mut depth = 0i32;
    let mut max_depth = 0i32;
    let mut in_string = false;
    let mut escape = false;

    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => {
                offsets.push((base + i + 1) as i32);
                stack.push((b, offsets.len() - 1));
                depth += 1;
                max_depth = max_depth.max(depth);
            }
            b'}' | b']' => {
                offsets.push((base + i + 1) as i32);
                let (open_b, open_idx) = stack.pop().ok_or(Error::UnbalancedBrackets)?;
                let expected = if b == b'}' { b'{' } else { b'[' };
                if open_b != expected {
                    return Err(Error::UnbalancedBrackets);
                }
                pairs.push((open_idx, offsets.len() - 1));
                depth -= 1;
            }
            b':' | b',' => {
                offsets.push((base + i + 1) as i32);
            }
            b'\n' if mark_newline => {
                offsets.push((base + i + 1) as i32);
            }
            _ => {}
        }
    }
    if in_string {
        return Err(Error::UnterminatedString);
    }
    if !stack.is_empty() {
        return Err(Error::UnbalancedBrackets);
    }
    Ok(RawStructure {
        offsets,
        pairs,
        depth: max_depth,
    })
}

/// JSON Lines: the kernel does not split the input by line at all — every
/// chunk (an implementation detail of GPU parallelism, see `tape/FORMAT.md`
/// §5's chunk-boundary note) is tokenized with the *same* structural bitmap
/// as Standard mode except that an unescaped `\n` outside a string is also
/// structural (`bitMapCreatorSimd` in `parse_json_lines.cu`, §1/§5). Bracket
/// pairing is unaffected: brackets never span a `\n` in valid per-line JSON,
/// so a single whole-input scan (stack returns to depth 0 at every `\n`)
/// produces the same pairing as the kernel's per-chunk pairing pass.
///
/// This single-pass scan is what reproduces the kernel exactly for the four
/// newline cases documented in `tape/FORMAT.md` §5: a trailing `\n` at EOF
/// and a blank line (`\n\n`) each get their own structural entry (no
/// skipping), a bare `\r` before `\n` is never itself structural, and chunk
/// boundaries (a CPU-builder-only non-concept) can't perturb the result
/// because they always fall immediately after a complete line in the real
/// loader (`load_file.cu`'s line-offset chunking).
fn build_lines(input: &[u8]) -> Result<RawStructure, Error> {
    scan_structural(input, 0, true)
}

/// Build a tape from `input`, matching `tape/FORMAT.md` byte for byte.
pub fn build_tape_cpu(input: &[u8], mode: Mode) -> Result<Tape, Error> {
    if input.is_empty() {
        return Err(Error::EmptyInput);
    }
    std::str::from_utf8(input).map_err(|_| Error::InvalidUtf8)?;

    let RawStructure {
        offsets,
        pairs,
        depth,
    } = match mode {
        Mode::Standard => scan_structural(input, 0, false)?,
        Mode::Lines => build_lines(input)?,
    };

    let n = offsets.len();
    let total = n + 2;
    let mut structural = vec![0i32; total];
    structural[1..=n].copy_from_slice(&offsets);
    structural[n + 1] = (n + 1) as i32;

    // pair_pos: -1 marks "undefined" for non-opener slots, matching the
    // fact that the GPU kernel never writes them (tape/FORMAT.md §3).
    let mut pair_pos = vec![-1i32; total];
    pair_pos[0] = (n + 1) as i32; // artificial open pairs with artificial close
    pair_pos[n + 1] = 0; // builder policy choice, see FORMAT.md §3 Unresolved #3
    for (a, b) in pairs {
        pair_pos[a + 1] = (b + 1) as i32;
    }

    Ok(Tape {
        structural: TapeStorage::from(structural),
        pair_pos: TapeStorage::from(pair_pos),
        depth,
    })
}
