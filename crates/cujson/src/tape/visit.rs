//! Single-pass event walk over a tape.

use std::ops::Range;

use super::document::{Document, Kind, scalar_kind, unescape_into};
use super::error::Error;

/// Receives the events of [`Document::visit`], in document order.
///
/// `string` and `key` get unescaped text (valid only for the call); `number` gets the raw, unparsed
/// bytes (cuJSON does not validate number grammar).
#[allow(unused_variables)]
pub trait Visitor {
    fn begin_object(&mut self) {}
    fn end_object(&mut self) {}
    fn begin_array(&mut self) {}
    fn end_array(&mut self) {}
    fn key(&mut self, key: &str) {}
    fn string(&mut self, value: &str) {}
    fn number(&mut self, raw: &[u8]) {}
    fn boolean(&mut self, value: bool) {}
    fn null(&mut self) {}
}

/// `raw` must lie inside a region the caller has checked to be valid UTF-8.
fn unquote<'a>(raw: &'a [u8], scratch: &'a mut String) -> Result<&'a str, Error> {
    let [b'"', inner @ .., b'"'] = raw else {
        return Err(Error::TypeMismatch("string"));
    };
    // `any` measured faster than `contains` on these short strings.
    #[allow(clippy::manual_contains)]
    let plain = !inner.iter().any(|&b| b == b'\\');
    if plain {
        // SAFETY: `inner` sits between two ASCII quote bytes of a region
        // checked to be valid UTF-8, so both of its ends are char boundaries.
        return Ok(unsafe { std::str::from_utf8_unchecked(inner) });
    }
    scratch.clear();
    unescape_into(inner, scratch, true)?;
    Ok(scratch.as_str())
}

fn scalar<V: Visitor>(raw: &[u8], v: &mut V, scratch: &mut String) -> Result<(), Error> {
    match scalar_kind(raw) {
        Kind::String => v.string(unquote(raw, scratch)?),
        Kind::Bool => match raw {
            b"true" => v.boolean(true),
            b"false" => v.boolean(false),
            _ => return Err(Error::TypeMismatch("bool")),
        },
        Kind::Null if raw == b"null" => v.null(),
        Kind::Null => return Err(Error::TypeMismatch("null")),
        _ => v.number(raw),
    }
    Ok(())
}

impl Document<'_> {
    /// Walk every value once, front to back, reading `structural` and the
    /// input sequentially. Unlike [`Node`](super::Node) navigation this
    /// never reads `pair_pos`. For a JSON Lines document the top-level
    /// values are visited in line order, blank lines skipped.
    pub fn visit<V: Visitor>(&self, v: &mut V) -> Result<(), Error> {
        let n = self.tape.structural.len();
        if n <= 2 {
            let raw = self.input.trim_ascii();
            let mut scratch = String::new();
            std::str::from_utf8(&self.input).map_err(|_| Error::InvalidUtf8)?;
            return if raw.is_empty() {
                Ok(())
            } else {
                scalar(raw, v, &mut scratch)
            };
        }
        self.visit_range(1..n - 1, v)
    }

    /// Split the tape into at most `parts` ranges of similar size, each a
    /// whole number of top-level values (lines), for [`visit_range`]
    /// on separate threads. Hops between lines with `pair_pos`, which
    /// is defined at openers on a GPU tape.
    ///
    /// [`visit_range`]: Document::visit_range
    pub fn split_lines(&self, parts: usize) -> Vec<Range<usize>> {
        let n = self.tape.structural.len();
        if n <= 2 {
            return std::iter::once(1..n - 1).collect();
        }
        let target = (n - 2).div_ceil(parts.max(1));
        let mut out = vec![];
        let mut start = 1usize;
        for delim in self.top_level_delims() {
            let next = delim + 1;
            if next < n - 1 && next - start >= target {
                out.push(start..next);
                start = next;
            }
        }
        out.push(start..n - 1);
        out
    }

    /// [`visit`](Document::visit) over the tape entries `range` (indices
    /// into `structural`, within `1..len-1`), as returned by
    /// [`split_lines`](Document::split_lines).
    pub fn visit_range<V: Visitor>(&self, range: Range<usize>, v: &mut V) -> Result<(), Error> {
        let mut scratch = String::new();
        let structural: &[i32] = &self.tape.structural;
        let input: &[u8] = &self.input;
        let n = structural.len();
        if range.start == 0 || range.end > n - 1 {
            return Err(Error::IndexOutOfRange);
        }
        let mut prev = if range.start == 1 {
            0
        } else {
            structural[range.start - 1] as usize
        };
        // Check the bytes this range reads once, so keys and strings need no
        // per-value check (see `unquote`).
        let region_end = if range.end == n - 1 {
            input.len()
        } else {
            (structural[range.end - 1] as usize).min(input.len())
        };
        if prev > region_end {
            return Err(Error::IndexOutOfRange);
        }
        std::str::from_utf8(&input[prev..region_end]).map_err(|_| Error::InvalidUtf8)?;
        let mut depth = 0usize;
        for &p in &structural[range.clone()] {
            let pos = (p - 1) as usize;
            // Every span must stay inside the region validated above.
            if pos < prev || pos >= region_end.max(1) {
                return Err(Error::IndexOutOfRange);
            }
            let raw = input[prev..pos].trim_ascii();
            match input[pos] {
                b'{' => {
                    v.begin_object();
                    depth += 1;
                }
                b'[' => {
                    v.begin_array();
                    depth += 1;
                }
                b':' => v.key(unquote(raw, &mut scratch)?),
                c @ (b',' | b'\n' | b'}' | b']') => {
                    if !raw.is_empty() {
                        scalar(raw, v, &mut scratch)?;
                    }
                    match c {
                        b'}' | b']' => {
                            depth = depth.checked_sub(1).ok_or(Error::UnbalancedBrackets)?;
                            if c == b'}' {
                                v.end_object();
                            } else {
                                v.end_array();
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
            prev = pos + 1;
        }
        if range.end == n - 1 {
            let raw = input[prev.min(input.len())..].trim_ascii();
            if !raw.is_empty() {
                scalar(raw, v, &mut scratch)?;
            }
        }
        if depth != 0 {
            return Err(Error::UnbalancedBrackets);
        }
        Ok(())
    }
}
