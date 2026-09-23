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
fn scan_structural(bytes: &[u8], base: usize) -> Result<RawStructure, Error> {
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

/// JSON Lines: split on raw `\n` bytes (safe because JSON forbids a literal
/// newline inside a string — any unescaped `\n` in valid input is a line
/// separator), scan each non-blank line independently, and splice in a
/// structural entry at each separating newline so it reads back as a comma
/// (`tape/FORMAT.md` §5, matching `getChar`'s `'\n' -> ','` translation).
fn build_lines(input: &[u8]) -> Result<RawStructure, Error> {
    let mut offsets = Vec::new();
    let mut pairs = Vec::new();
    let mut max_depth = 0i32;
    let mut cursor = 0usize;
    let mut pending_newline: Option<i32> = None;
    let n = input.len();

    loop {
        let rest = &input[cursor..];
        let (line_end, next_cursor, nl_abs) = match rest.iter().position(|&b| b == b'\n') {
            Some(r) => (cursor + r, cursor + r + 1, Some(cursor + r)),
            None => (n, n + 1, None),
        };
        let line = &input[cursor..line_end];
        if !line.iter().all(u8::is_ascii_whitespace) {
            if let Some(pending) = pending_newline.take() {
                offsets.push(pending);
            }
            let scan = scan_structural(line, cursor)?;
            let index_base = offsets.len();
            for &(a, b) in &scan.pairs {
                pairs.push((index_base + a, index_base + b));
            }
            offsets.extend(scan.offsets);
            max_depth = max_depth.max(scan.depth);
        }
        if let Some(nl) = nl_abs {
            pending_newline = Some((nl + 1) as i32);
        }
        if next_cursor > n {
            break;
        }
        cursor = next_cursor;
    }
    Ok(RawStructure {
        offsets,
        pairs,
        depth: max_depth,
    })
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
        Mode::Standard => scan_structural(input, 0)?,
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
