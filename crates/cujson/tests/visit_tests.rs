//! `Document::visit` must emit exactly the events `Node` navigation yields.
#![cfg(feature = "cpu-reference")]

use std::borrow::Cow;
use std::path::Path;

use cujson::tape::{Document, Kind, Mode, Node, Visitor, build_tape_cpu};

#[derive(Default)]
struct Recorder(Vec<String>);

impl Visitor for Recorder {
    fn begin_object(&mut self) {
        self.0.push("{".into());
    }
    fn end_object(&mut self) {
        self.0.push("}".into());
    }
    fn begin_array(&mut self) {
        self.0.push("[".into());
    }
    fn end_array(&mut self) {
        self.0.push("]".into());
    }
    fn key(&mut self, key: &str) {
        self.0.push(format!("k:{key}"));
    }
    fn string(&mut self, value: &str) {
        self.0.push(format!("s:{value}"));
    }
    fn number(&mut self, raw: &[u8]) {
        self.0.push(format!("n:{}", String::from_utf8_lossy(raw)));
    }
    fn boolean(&mut self, value: bool) {
        self.0.push(format!("b:{value}"));
    }
    fn null(&mut self) {
        self.0.push("z".into());
    }
}

fn node_events(n: Node<'_>, out: &mut Vec<String>) {
    match n.kind() {
        Kind::Object => {
            out.push("{".into());
            for (k, c) in n.iter_object().unwrap() {
                out.push(format!("k:{k}"));
                node_events(c, out);
            }
            out.push("}".into());
        }
        Kind::Array => {
            out.push("[".into());
            for c in n.iter_array().unwrap() {
                node_events(c, out);
            }
            out.push("]".into());
        }
        Kind::String => out.push(format!("s:{}", n.as_str().unwrap())),
        Kind::Number => out.push(format!("n:{}", String::from_utf8_lossy(n.raw()))),
        Kind::Bool => out.push(format!("b:{}", n.as_bool().unwrap())),
        Kind::Null => out.push("z".into()),
    }
}

fn check(bytes: &[u8], mode: Mode) {
    let tape = build_tape_cpu(bytes, mode).expect("tape");
    let doc = Document::new(Cow::Borrowed(bytes), tape);
    let mut want = vec![];
    match mode {
        Mode::Standard => node_events(doc.root(), &mut want),
        Mode::Lines => doc.lines().for_each(|n| node_events(n, &mut want)),
    }
    let mut got = Recorder::default();
    doc.visit(&mut got).expect("visit");
    assert_eq!(got.0, want, "{}", String::from_utf8_lossy(bytes));
    for parts in [1, 2, 3, 7, 64] {
        let mut joined = Recorder::default();
        let ranges = doc.split_lines(parts);
        assert!(ranges.len() <= parts.max(1));
        for r in ranges {
            doc.visit_range(r, &mut joined).expect("visit_range");
        }
        assert_eq!(
            joined.0,
            want,
            "parts={parts}: {}",
            String::from_utf8_lossy(bytes)
        );
    }
}

#[test]
fn standard_edge_cases() {
    for j in [
        "{}",
        "[]",
        "[null]",
        "[[]]",
        r#"{"a":{}}"#,
        r#"{"a":[{}],"b":[[],[null],[1]]}"#,
        r#"{"a b":"x\"yé\\","c":[1,-2.5e3,true,false,null,[],{}]}"#,
        "  {\n  \"a\" : [ 1 ,\n 2 ] ,\n \"b\" : { } \n}  ",
        "42",
        " \"s\" ",
        "null",
    ] {
        check(j.as_bytes(), Mode::Standard);
    }
}

#[test]
fn lines_edge_cases() {
    for j in [
        "{}\n[]\n\n{\"a\":1}\n7\n\"x\"\n",
        "{\"a\":[1,2]}\n{\"b\":null}",
        "1\n2\n3",
        "[null]\n[]\n",
    ] {
        check(j.as_bytes(), Mode::Lines);
    }
}

#[test]
fn fixtures() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    check(
        &std::fs::read(dir.join("twitter_sample_large_record.json")).unwrap(),
        Mode::Standard,
    );
    check(
        &std::fs::read(dir.join("twitter_sample_small_records.json")).unwrap(),
        Mode::Lines,
    );
}

#[test]
fn mismatched_tape_and_input_does_not_panic() {
    let tape = build_tape_cpu(b"[1]", Mode::Standard).unwrap();
    let doc = Document::new(Cow::Borrowed(&b"[1]]"[..]), tape);
    let _ = doc.visit(&mut Recorder::default());
}

#[test]
fn invalid_utf8_in_a_string_is_an_error() {
    let good = "{\"a\":\"\u{e9}x\"}".as_bytes().to_vec();
    let tape = build_tape_cpu(&good, Mode::Standard).unwrap();
    let mut bad = good.clone();
    let at = bad.iter().position(|&b| b == 0xC3).unwrap();
    bad[at + 1] = b'('; // 0xC3 0x28 is not a valid sequence
    let doc = Document::new(Cow::Owned(bad), tape);
    assert_eq!(
        doc.visit(&mut Recorder::default()),
        Err(cujson::tape::Error::InvalidUtf8)
    );
}

#[test]
#[allow(irrefutable_let_patterns)] // `Pinned` exists only with the `cuda` feature
fn corrupt_tape_offsets_are_an_error_not_a_read_outside_the_region() {
    let input = br#"{"a":"xyz","b":[1,2]}"#;
    for damage in 0..8 {
        let mut tape = build_tape_cpu(input, Mode::Standard).unwrap();
        let cujson::tape::TapeStorage::Owned(s) = &mut tape.structural else {
            unreachable!()
        };
        let n = s.len();
        s[1 + damage % (n - 2)] = if damage % 2 == 0 { 1 } else { 1000 };
        let doc = Document::new(Cow::Borrowed(&input[..]), tape);
        let _ = doc.visit(&mut Recorder::default());
        for r in doc.split_lines(3) {
            let _ = doc.visit_range(r, &mut Recorder::default());
        }
    }
}
