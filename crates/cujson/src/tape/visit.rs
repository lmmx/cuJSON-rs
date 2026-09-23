//! Single-pass event walk over a tape.

use std::borrow::Cow;

use super::document::{Document, Kind, scalar_kind, unescape};
use super::error::Error;

/// Receives the events of [`Document::visit`], in document order.
///
/// `string` and `key` get unescaped text; `number` gets the raw, unparsed
/// bytes (cuJSON does not validate number grammar).
#[allow(unused_variables)]
pub trait Visitor {
    fn begin_object(&mut self) {}
    fn end_object(&mut self) {}
    fn begin_array(&mut self) {}
    fn end_array(&mut self) {}
    fn key(&mut self, key: Cow<'_, str>) {}
    fn string(&mut self, value: Cow<'_, str>) {}
    fn number(&mut self, raw: &[u8]) {}
    fn boolean(&mut self, value: bool) {}
    fn null(&mut self) {}
}

fn unquote(raw: &[u8]) -> Result<Cow<'_, str>, Error> {
    match raw {
        [b'"', inner @ .., b'"'] => unescape(inner),
        _ => Err(Error::TypeMismatch("string")),
    }
}

fn scalar<V: Visitor>(raw: &[u8], v: &mut V) -> Result<(), Error> {
    match scalar_kind(raw) {
        Kind::String => v.string(unquote(raw)?),
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
        let structural: &[i32] = &self.tape.structural;
        let input: &[u8] = &self.input;
        let n = structural.len();
        if n <= 2 {
            let raw = input.trim_ascii();
            return if raw.is_empty() {
                Ok(())
            } else {
                scalar(raw, v)
            };
        }
        let mut prev = 0usize;
        let mut depth = 0usize;
        for &p in &structural[1..n - 1] {
            let pos = (p - 1) as usize;
            let raw = input[prev..pos.max(prev)].trim_ascii();
            match input[pos] {
                b'{' => {
                    v.begin_object();
                    depth += 1;
                }
                b'[' => {
                    v.begin_array();
                    depth += 1;
                }
                b':' => v.key(unquote(raw)?),
                c @ (b',' | b'\n' | b'}' | b']') => {
                    if !raw.is_empty() {
                        scalar(raw, v)?;
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
        let raw = input[prev.min(input.len())..].trim_ascii();
        if !raw.is_empty() {
            scalar(raw, v)?;
        }
        if depth != 0 {
            return Err(Error::UnbalancedBrackets);
        }
        Ok(())
    }
}
