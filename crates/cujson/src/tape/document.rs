//! `Document`/`Node` navigator over a `Tape` (`tape/FORMAT.md`).

use std::borrow::Cow;

use super::error::Error;
use super::storage::Tape;

#[cfg(test)]
use super::storage::TapeStorage;

/// The kind of JSON value a `Node` represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Object,
    Array,
    String,
    Number,
    Bool,
    Null,
}

/// A parsed document: the raw input bytes plus its tape.
#[derive(Debug, Clone)]
pub struct Document<'a> {
    pub input: Cow<'a, [u8]>,
    pub tape: Tape,
}

#[derive(Debug, Clone, Copy)]
enum NodeRepr {
    Container {
        open: usize,
        close: usize,
        kind: Kind,
    },
    // Half-open span (lo, hi) of *tape indices* delimiting a scalar: the
    // scalar's bytes lie strictly between structural entries lo and hi.
    Scalar {
        lo: usize,
        hi: usize,
    },
}

/// A value within a `Document`'s tape: an object, array, or scalar.
#[derive(Debug, Clone, Copy)]
pub struct Node<'d> {
    doc: &'d Document<'d>,
    repr: NodeRepr,
}

impl<'a> Document<'a> {
    pub fn new(input: Cow<'a, [u8]>, tape: Tape) -> Self {
        Document { input, tape }
    }

    fn total(&self) -> usize {
        self.tape.structural.len()
    }

    /// 0-based byte position corresponding to tape index `idx`, or a
    /// sentinel one-before-the-start / one-past-the-end for the two
    /// artificial wrapper entries (`tape/FORMAT.md` §2).
    fn byte_pos(&self, idx: usize) -> isize {
        if idx == 0 {
            -1
        } else if idx + 1 == self.total() {
            self.input.len() as isize
        } else {
            (self.tape.structural[idx] - 1) as isize
        }
    }

    /// Mirrors the upstream `getChar` (`query_iterator_standard_json.cpp:148-165`).
    fn get_char(&self, idx: usize) -> u8 {
        if idx >= self.total() {
            return 0;
        }
        if idx == 0 {
            return b'[';
        }
        if idx + 1 == self.total() {
            return b']';
        }
        let pos = self.tape.structural[idx] - 1;
        if pos < 0 || pos as usize >= self.input.len() {
            return 0;
        }
        let c = self.input[pos as usize];
        if c == b'\n' { b',' } else { c }
    }

    fn pair_pos(&self, idx: usize) -> usize {
        self.tape.pair_pos[idx].max(0) as usize
    }

    fn scalar_bytes(&self, lo: usize, hi: usize) -> &[u8] {
        let start = (self.byte_pos(lo) + 1) as usize;
        let end = self.byte_pos(hi) as usize;
        let end = end.max(start);
        self.input[start..end].trim_ascii()
    }

    /// Read the value that follows structural token `delim` (a `[`, `{`,
    /// `,`, or `:` tape index), mirroring the upstream `getValue`'s
    /// dispatch on `currentNodeChar` (`query_iterator_standard_json.cpp:562-604`).
    fn read_value(&'a self, delim: usize) -> Node<'a> {
        let peek = self.get_char(delim + 1);
        if peek == b'{' || peek == b'[' {
            let open = delim + 1;
            let close = self.pair_pos(open);
            let kind = if peek == b'{' {
                Kind::Object
            } else {
                Kind::Array
            };
            Node {
                doc: self,
                repr: NodeRepr::Container { open, close, kind },
            }
        } else {
            Node {
                doc: self,
                repr: NodeRepr::Scalar {
                    lo: delim,
                    hi: delim + 1,
                },
            }
        }
    }

    /// The document's first top-level value. For a JSON Lines document this
    /// is the first line only — use `lines()` to see the rest.
    pub fn root(&'a self) -> Node<'a> {
        self.read_value(0)
    }

    /// Iterate the top-level values of a JSON Lines document (or the single
    /// root value of a standard document, as a one-element iterator).
    ///
    /// The tape itself marks *every* unescaped `\n` structural, matching the
    /// kernel exactly (`tape/FORMAT.md` §5): a trailing `\n` at EOF and each
    /// `\n` of a blank line each get their own structural entry, read back
    /// as `,` by `get_char`. That can put two comma-like separators back to
    /// back with nothing between them. This iterator — a navigator-layer
    /// convenience, not part of the tape format itself — skips the
    /// resulting empty scalar spans rather than yielding a phantom
    /// zero-length "value", so blank lines and a trailing newline are
    /// silently absorbed instead of appearing as bogus documents.
    pub fn lines(&'a self) -> impl Iterator<Item = Node<'a>> + 'a {
        children_iter(self, 0, self.total() - 1).filter(|node| {
            !matches!(node.repr, NodeRepr::Scalar { lo, hi } if node.doc.scalar_bytes(lo, hi).is_empty())
        })
    }

    /// Resolve an RFC 6901 JSON Pointer against the root value.
    pub fn pointer(&'a self, pointer: &str) -> Option<Node<'a>> {
        let mut node = self.root();
        if pointer.is_empty() {
            return Some(node);
        }
        if !pointer.starts_with('/') {
            return None;
        }
        for raw_tok in pointer[1..].split('/') {
            let tok = raw_tok.replace("~1", "/").replace("~0", "~");
            node = match node.kind() {
                Kind::Object => node.get(&tok)?,
                Kind::Array => {
                    let idx: usize = tok.parse().ok()?;
                    node.index(idx)?
                }
                _ => return None,
            };
        }
        Some(node)
    }
}

fn scalar_kind(bytes: &[u8]) -> Kind {
    match bytes.first() {
        Some(b'"') => Kind::String,
        Some(b't') | Some(b'f') => Kind::Bool,
        Some(b'n') => Kind::Null,
        _ => Kind::Number,
    }
}

/// Shared child-walking loop for both arrays and (key-stripped) objects:
/// yields successive values found after each delimiter, starting at `open`
/// and stopping at `close`. See the walkthrough in this module's tests.
fn children_iter<'a>(
    doc: &'a Document<'a>,
    open: usize,
    close: usize,
) -> impl Iterator<Item = Node<'a>> + 'a {
    let mut delim = open;
    // A tape-adjacent open/close (no comma between them) is ambiguous: it
    // means either an empty container or a single scalar with no comma
    // (e.g. `[null]` has no comma structural entry either). Disambiguate on
    // actual byte content instead of tape-index adjacency.
    let mut empty = doc.scalar_bytes(open, close).is_empty();
    std::iter::from_fn(move || {
        if empty || delim == close {
            empty = true;
            return None;
        }
        let node = doc.read_value(delim);
        delim = match node.repr {
            NodeRepr::Container { close: c, .. } => c + 1,
            NodeRepr::Scalar { hi, .. } => hi,
        };
        Some(node)
    })
}

fn object_pairs<'a>(
    doc: &'a Document<'a>,
    open: usize,
    close: usize,
) -> impl Iterator<Item = (Cow<'a, str>, Node<'a>)> + 'a {
    let mut delim = open;
    let mut empty = doc.scalar_bytes(open, close).is_empty();
    std::iter::from_fn(move || {
        if empty || delim == close {
            empty = true;
            return None;
        }
        let colon_idx = delim + 1;
        let key_bytes = doc.scalar_bytes(delim, colon_idx);
        let key =
            if key_bytes.len() >= 2 && key_bytes[0] == b'"' && *key_bytes.last().unwrap() == b'"' {
                unescape(&key_bytes[1..key_bytes.len() - 1]).unwrap_or(Cow::Borrowed(""))
            } else {
                Cow::Borrowed("")
            };
        let node = doc.read_value(colon_idx);
        delim = match node.repr {
            NodeRepr::Container { close: c, .. } => c + 1,
            NodeRepr::Scalar { hi, .. } => hi,
        };
        Some((key, node))
    })
}

fn parse_hex4(bytes: &[u8], start: usize) -> Result<u32, Error> {
    let s = bytes.get(start..start + 4).ok_or(Error::InvalidEscape)?;
    let s = std::str::from_utf8(s).map_err(|_| Error::InvalidEscape)?;
    u32::from_str_radix(s, 16).map_err(|_| Error::InvalidEscape)
}

fn unescape(bytes: &[u8]) -> Result<Cow<'_, str>, Error> {
    if !bytes.contains(&b'\\') {
        return std::str::from_utf8(bytes)
            .map(Cow::Borrowed)
            .map_err(|_| Error::InvalidUtf8);
    }
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            let start = i;
            while i < bytes.len() && bytes[i] != b'\\' {
                i += 1;
            }
            out.push_str(std::str::from_utf8(&bytes[start..i]).map_err(|_| Error::InvalidUtf8)?);
            continue;
        }
        let e = *bytes.get(i + 1).ok_or(Error::InvalidEscape)?;
        match e {
            b'"' => {
                out.push('"');
                i += 2;
            }
            b'\\' => {
                out.push('\\');
                i += 2;
            }
            b'/' => {
                out.push('/');
                i += 2;
            }
            b'b' => {
                out.push('\u{8}');
                i += 2;
            }
            b'f' => {
                out.push('\u{c}');
                i += 2;
            }
            b'n' => {
                out.push('\n');
                i += 2;
            }
            b'r' => {
                out.push('\r');
                i += 2;
            }
            b't' => {
                out.push('\t');
                i += 2;
            }
            b'u' => {
                let cp = parse_hex4(bytes, i + 2)?;
                if (0xD800..=0xDBFF).contains(&cp) {
                    if bytes.get(i + 6) == Some(&b'\\') && bytes.get(i + 7) == Some(&b'u') {
                        let low = parse_hex4(bytes, i + 8)?;
                        if (0xDC00..=0xDFFF).contains(&low) {
                            let c = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                            out.push(char::from_u32(c).ok_or(Error::InvalidEscape)?);
                            i += 12;
                        } else {
                            return Err(Error::InvalidEscape);
                        }
                    } else {
                        return Err(Error::InvalidEscape);
                    }
                } else {
                    out.push(char::from_u32(cp).ok_or(Error::InvalidEscape)?);
                    i += 6;
                }
            }
            _ => return Err(Error::InvalidEscape),
        }
    }
    Ok(Cow::Owned(out))
}

impl<'d> Node<'d> {
    pub fn kind(&self) -> Kind {
        match self.repr {
            NodeRepr::Container { kind, .. } => kind,
            NodeRepr::Scalar { lo, hi } => scalar_kind(self.doc.scalar_bytes(lo, hi)),
        }
    }

    /// Number of children, for object/array nodes only.
    pub fn len(&self) -> Option<usize> {
        match self.repr {
            NodeRepr::Container {
                open,
                close,
                kind: Kind::Object,
            } => Some(object_pairs(self.doc, open, close).count()),
            NodeRepr::Container { open, close, .. } => {
                Some(children_iter(self.doc, open, close).count())
            }
            NodeRepr::Scalar { .. } => None,
        }
    }

    pub fn is_empty(&self) -> Option<bool> {
        self.len().map(|n| n == 0)
    }

    pub fn iter_array(&self) -> Option<impl Iterator<Item = Node<'d>> + 'd> {
        match self.repr {
            NodeRepr::Container {
                open,
                close,
                kind: Kind::Array,
            } => Some(children_iter(self.doc, open, close)),
            _ => None,
        }
    }

    pub fn iter_object(&self) -> Option<impl Iterator<Item = (Cow<'d, str>, Node<'d>)> + 'd> {
        match self.repr {
            NodeRepr::Container {
                open,
                close,
                kind: Kind::Object,
            } => Some(object_pairs(self.doc, open, close)),
            _ => None,
        }
    }

    pub fn index(&self, i: usize) -> Option<Node<'d>> {
        self.iter_array()?.nth(i)
    }

    pub fn get(&self, key: &str) -> Option<Node<'d>> {
        self.iter_object()?.find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// The value's exact input bytes (whitespace-trimmed for a scalar,
    /// bracket-to-bracket inclusive for a container).
    pub fn raw(&self) -> &'d [u8] {
        match self.repr {
            NodeRepr::Container { open, close, .. } => {
                let start = self.doc.byte_pos(open) as usize;
                let end = (self.doc.byte_pos(close) + 1) as usize;
                &self.doc.input[start..end]
            }
            NodeRepr::Scalar { lo, hi } => self.doc.scalar_bytes(lo, hi),
        }
    }

    pub fn as_str(&self) -> Result<Cow<'d, str>, Error> {
        match self.repr {
            NodeRepr::Scalar { lo, hi } => {
                let raw = self.doc.scalar_bytes(lo, hi);
                if raw.len() < 2 || raw[0] != b'"' || *raw.last().unwrap() != b'"' {
                    return Err(Error::TypeMismatch("string"));
                }
                unescape(&raw[1..raw.len() - 1])
            }
            _ => Err(Error::TypeMismatch("string")),
        }
    }

    pub fn as_f64(&self) -> Result<f64, Error> {
        match self.repr {
            NodeRepr::Scalar { lo, hi } => {
                let raw = self.doc.scalar_bytes(lo, hi);
                std::str::from_utf8(raw)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .ok_or(Error::InvalidNumber)
            }
            _ => Err(Error::TypeMismatch("number")),
        }
    }

    pub fn as_i64(&self) -> Result<i64, Error> {
        match self.repr {
            NodeRepr::Scalar { lo, hi } => {
                let raw = self.doc.scalar_bytes(lo, hi);
                std::str::from_utf8(raw)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .ok_or(Error::InvalidNumber)
            }
            _ => Err(Error::TypeMismatch("number")),
        }
    }

    pub fn as_bool(&self) -> Result<bool, Error> {
        match self.repr {
            NodeRepr::Scalar { lo, hi } => match self.doc.scalar_bytes(lo, hi) {
                b"true" => Ok(true),
                b"false" => Ok(false),
                _ => Err(Error::TypeMismatch("bool")),
            },
            _ => Err(Error::TypeMismatch("bool")),
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self.repr, NodeRepr::Scalar { lo, hi } if self.doc.scalar_bytes(lo, hi) == b"null")
    }
}

#[cfg(feature = "serde")]
impl<'d> Node<'d> {
    pub fn to_value(&self) -> serde_json::Value {
        use serde_json::Value;
        match self.kind() {
            Kind::Null => Value::Null,
            Kind::Bool => Value::Bool(self.as_bool().unwrap_or(false)),
            Kind::Number => {
                // Parse with std's `f64`/`i64` parsers rather than
                // `serde_json::from_str`: serde_json's own float lexer can
                // be off by one ULP for some inputs, which broke exact
                // round-tripping in the property tests.
                let raw = std::str::from_utf8(self.raw()).unwrap_or("0");
                if !raw.contains(['.', 'e', 'E']) {
                    if let Ok(i) = raw.parse::<i64>() {
                        Value::Number(i.into())
                    } else if let Ok(u) = raw.parse::<u64>() {
                        Value::Number(u.into())
                    } else {
                        raw.parse::<f64>()
                            .ok()
                            .and_then(serde_json::Number::from_f64)
                            .map(Value::Number)
                            .unwrap_or(Value::Null)
                    }
                } else {
                    raw.parse::<f64>()
                        .ok()
                        .and_then(serde_json::Number::from_f64)
                        .map(Value::Number)
                        .unwrap_or(Value::Null)
                }
            }
            Kind::String => {
                Value::String(self.as_str().map(|c| c.into_owned()).unwrap_or_default())
            }
            Kind::Array => Value::Array(self.iter_array().unwrap().map(|n| n.to_value()).collect()),
            Kind::Object => Value::Object(
                self.iter_object()
                    .unwrap()
                    .map(|(k, v)| (k.into_owned(), v.to_value()))
                    .collect(),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-written tape for `{"a":1,"b":[2,3]}`, worked out by hand from
    /// `tape/FORMAT.md` §1-3 rather than produced by `build_tape_cpu`, so
    /// these tests exercise the navigator independently of the builder.
    fn small_object_doc() -> Document<'static> {
        let input: &'static [u8] = b"{\"a\":1,\"b\":[2,3]}";
        // real structural bytes (1-based): { : , : [ , ]  }
        //                        offset:   1 5 7 11 12 14 16 17
        let structural = vec![0, 1, 5, 7, 11, 12, 14, 16, 17, 9];
        let mut pair_pos = vec![-1i32; 10];
        pair_pos[0] = 9;
        pair_pos[9] = 0;
        pair_pos[1] = 8; // '{' (tape idx 1) pairs with '}' (tape idx 8)
        pair_pos[5] = 7; // '[' (tape idx 5) pairs with ']' (tape idx 7)
        let tape = Tape {
            structural: TapeStorage::from(structural),
            pair_pos: TapeStorage::from(pair_pos),
            depth: 2,
        };
        Document::new(Cow::Borrowed(input), tape)
    }

    #[test]
    fn root_is_object() {
        let doc = small_object_doc();
        assert_eq!(doc.root().kind(), Kind::Object);
        assert_eq!(doc.root().len(), Some(2));
    }

    #[test]
    fn get_scalar_field() {
        let doc = small_object_doc();
        let a = doc.root().get("a").unwrap();
        assert_eq!(a.kind(), Kind::Number);
        assert_eq!(a.as_i64(), Ok(1));
    }

    #[test]
    fn get_array_field_and_index() {
        let doc = small_object_doc();
        let b = doc.root().get("b").unwrap();
        assert_eq!(b.kind(), Kind::Array);
        assert_eq!(b.len(), Some(2));
        assert_eq!(b.index(0).unwrap().as_i64(), Ok(2));
        assert_eq!(b.index(1).unwrap().as_i64(), Ok(3));
        assert!(b.index(2).is_none());
    }

    #[test]
    fn missing_key_is_none() {
        let doc = small_object_doc();
        assert!(doc.root().get("nope").is_none());
    }

    #[test]
    fn pointer_navigates_nested_array() {
        let doc = small_object_doc();
        assert_eq!(doc.pointer("/b/1").unwrap().as_i64(), Ok(3));
        assert_eq!(doc.pointer("").unwrap().kind(), Kind::Object);
        assert!(doc.pointer("/does/not/exist").is_none());
    }

    #[test]
    fn string_unescaping() {
        // {"k":"a\nbA\"c"} — the string value contains the two-character
        // escapes \n and \" literally (not a real newline byte).
        let input: &'static [u8] = br#"{"k":"a\nbA\"c"}"#;
        // structural bytes (1-based offsets): { at 1, : at 5, } at 16
        let structural = vec![0, 1, 5, 16, 4];
        let pair_pos = vec![4, 3, -1, -1, 0];
        let tape = Tape {
            structural: TapeStorage::from(structural),
            pair_pos: TapeStorage::from(pair_pos),
            depth: 1,
        };
        let doc = Document::new(Cow::Borrowed(input), tape);
        let v = doc.root().get("k").unwrap();
        assert_eq!(v.kind(), Kind::String);
        assert_eq!(v.as_str().unwrap(), "a\nbA\"c");
    }
}
