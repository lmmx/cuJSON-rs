//! One checksum, computed identically over either parser's output. It visits
//! every node, unescapes every string and key, and parses every number, so
//! neither parser can skip work the other does. Object entries are combined
//! commutatively because `simd_json`'s object iteration order is not the
//! document order.

use cujson::tape::{Kind, Node, Visitor};
use simd_json::{BorrowedValue as V, StaticNode as S};

const TAG_NULL: u64 = 0x11;
const TAG_BOOL: u64 = 0x22;
const TAG_INT: u64 = 0x33;
const TAG_FLOAT: u64 = 0x44;
const TAG_STR: u64 = 0x55;
const TAG_ARR: u64 = 0x66;
const TAG_OBJ: u64 = 0x77;

#[inline]
fn mix(h: u64, x: u64) -> u64 {
    (h.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95)
}

#[inline]
fn hash_bytes(b: &[u8]) -> u64 {
    let mut h = b.len() as u64;
    let (chunks, rem) = b.as_chunks::<8>();
    for c in chunks {
        h = mix(h, u64::from_le_bytes(*c));
    }
    let mut tail = [0u8; 8];
    tail[..rem.len()].copy_from_slice(rem);
    mix(h, u64::from_le_bytes(tail))
}

#[inline]
fn int(i: i64) -> u64 {
    mix(TAG_INT, i as u64)
}

#[inline]
fn float(f: f64) -> u64 {
    mix(TAG_FLOAT, f.to_bits())
}

pub fn walk_simd(v: &V) -> u64 {
    match v {
        V::Static(S::Null) => TAG_NULL,
        V::Static(S::Bool(b)) => mix(TAG_BOOL, *b as u64),
        V::Static(S::I64(i)) => int(*i),
        V::Static(S::U64(u)) => match i64::try_from(*u) {
            Ok(i) => int(i),
            Err(_) => float(*u as f64),
        },
        V::Static(S::F64(f)) => float(*f),
        V::String(s) => mix(TAG_STR, hash_bytes(s.as_bytes())),
        V::Array(a) => mix(
            a.iter().fold(TAG_ARR, |h, x| mix(h, walk_simd(x))),
            a.len() as u64,
        ),
        V::Object(o) => {
            let mut acc = 0u64;
            for (k, x) in o.iter() {
                acc = acc.wrapping_add(mix(hash_bytes(k.as_bytes()), walk_simd(x)));
            }
            mix(mix(TAG_OBJ, o.len() as u64), acc)
        }
    }
}

pub fn walk_cujson(n: Node<'_>) -> u64 {
    match n.kind() {
        Kind::Null => TAG_NULL,
        Kind::Bool => mix(TAG_BOOL, n.as_bool().expect("bool") as u64),
        Kind::Number => match n.as_i64() {
            Ok(i) => int(i),
            Err(_) => float(n.as_f64().expect("number")),
        },
        Kind::String => mix(TAG_STR, hash_bytes(n.as_str().expect("string").as_bytes())),
        Kind::Array => {
            let mut h = TAG_ARR;
            let mut len = 0u64;
            for x in n.iter_array().expect("array") {
                h = mix(h, walk_cujson(x));
                len += 1;
            }
            mix(h, len)
        }
        Kind::Object => {
            let mut acc = 0u64;
            let mut len = 0u64;
            for (k, x) in n.iter_object().expect("object") {
                acc = acc.wrapping_add(mix(hash_bytes(k.as_bytes()), walk_cujson(x)));
                len += 1;
            }
            mix(mix(TAG_OBJ, len), acc)
        }
    }
}

struct Frame {
    obj: bool,
    acc: u64,
    len: u64,
    key: u64,
}

/// Same checksum as `walk_cujson`, computed from `Document::visit` events.
#[derive(Default)]
pub struct HashVisitor {
    stack: Vec<Frame>,
    pub rows: u64,
    pub hash: u64,
}

impl HashVisitor {
    fn feed(&mut self, h: u64) {
        match self.stack.last_mut() {
            Some(f) if f.obj => {
                f.acc = f.acc.wrapping_add(mix(f.key, h));
                f.len += 1;
            }
            Some(f) => {
                f.acc = mix(f.acc, h);
                f.len += 1;
            }
            None => {
                self.hash = self.hash.wrapping_add(h);
                self.rows += 1;
            }
        }
    }

    fn end(&mut self) {
        let f = self.stack.pop().expect("balanced");
        let h = if f.obj {
            mix(mix(TAG_OBJ, f.len), f.acc)
        } else {
            mix(f.acc, f.len)
        };
        self.feed(h);
    }
}

impl Visitor for HashVisitor {
    fn begin_object(&mut self) {
        self.stack.push(Frame {
            obj: true,
            acc: 0,
            len: 0,
            key: 0,
        });
    }
    fn begin_array(&mut self) {
        self.stack.push(Frame {
            obj: false,
            acc: TAG_ARR,
            len: 0,
            key: 0,
        });
    }
    fn end_object(&mut self) {
        self.end();
    }
    fn end_array(&mut self) {
        self.end();
    }
    fn key(&mut self, key: &str) {
        self.stack.last_mut().expect("key in object").key = hash_bytes(key.as_bytes());
    }
    fn string(&mut self, value: &str) {
        self.feed(mix(TAG_STR, hash_bytes(value.as_bytes())));
    }
    fn number(&mut self, raw: &[u8]) {
        let s = std::str::from_utf8(raw).expect("utf8 number");
        self.feed(match s.parse::<i64>() {
            Ok(i) => int(i),
            Err(_) => float(s.parse().expect("number")),
        });
    }
    fn boolean(&mut self, value: bool) {
        self.feed(mix(TAG_BOOL, value as u64));
    }
    fn null(&mut self) {
        self.feed(TAG_NULL);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agree(json: &str) {
        let mut bytes = json.as_bytes().to_vec();
        let simd = walk_simd(&simd_json::to_borrowed_value(&mut bytes).unwrap());
        let doc = cujson::cpu::parse(json.as_bytes(), cujson::cpu::Mode::Standard).unwrap();
        assert_eq!(simd, walk_cujson(doc.root()), "{json}");
    }

    #[test]
    fn parsers_agree_on_edge_cases() {
        for j in [
            r#"{"a":[1,2.5,-0,1e2,1.0,9223372036854775807,9223372036854775808,-9223372036854775807],"b":{}}"#,
            r#"{"s":"café 😀 \"q\" \\ \n \/","":[],"kéy":null}"#,
            r#"[true,false,null,[],{},[[]],{"a":{"b":{"c":[{}]}}}]"#,
            r#"{"z":1,"a":2,"m":{"y":1,"x":2}}"#,
            r#"{"lat":51.50722,"lon":-0.1275,"prec":1.0E-05}"#,
        ] {
            agree(j);
        }
    }

    #[test]
    fn visitor_checksum_matches_node_walk() {
        let j = r#"{"a":[1,2.5,-0,1e2,true,null,"é\n"],"b":{"c":[],"d":{}},"":[[1],[]]}"#;
        let d = cujson::cpu::parse(j.as_bytes(), cujson::cpu::Mode::Standard).unwrap();
        let mut v = HashVisitor::default();
        d.visit(&mut v).unwrap();
        assert_eq!((v.rows, v.hash), (1, walk_cujson(d.root())));
    }

    #[test]
    fn checksum_detects_differences() {
        let h = |j: &str| {
            let d = cujson::cpu::parse(j.as_bytes(), cujson::cpu::Mode::Standard).unwrap();
            walk_cujson(d.root())
        };
        assert_ne!(h(r#"{"a":1}"#), h(r#"{"a":2}"#));
        assert_ne!(h(r#"{"a":1}"#), h(r#"{"b":1}"#));
        assert_ne!(h(r#"[1,2]"#), h(r#"[2,1]"#));
        assert_ne!(h(r#"{"a":1}"#), h(r#"{"a":1.5}"#));
        assert_eq!(h(r#"{"a":1,"b":2}"#), h(r#"{"b":2,"a":1}"#));
    }
}
