//! Tier 1 tests for the CPU reference tape builder and navigator.
//!
//! Run with: `cargo test -p cujson --features cpu-reference,serde`. Gated on
//! both features so `cargo test --workspace` (no features) still compiles
//! this binary, just with nothing in it.
#![cfg(all(feature = "cpu-reference", feature = "serde"))]

use std::borrow::Cow;
use std::path::Path;

use cujson::tape::{Document, Mode, build_tape_cpu};
use proptest::prelude::*;

fn doc_from<'a>(bytes: &'a [u8], mode: Mode) -> Document<'a> {
    let tape = build_tape_cpu(bytes, mode).expect("build_tape_cpu");
    Document::new(Cow::Borrowed(bytes), tape)
}

// ---------------------------------------------------------------------
// Fixture tests
// ---------------------------------------------------------------------

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

#[test]
fn fixture_large_record_matches_serde_json() {
    let path = fixtures_dir().join("twitter_sample_large_record.json");
    let bytes = std::fs::read(&path).unwrap();
    let expected: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let doc = doc_from(&bytes, Mode::Standard);
    let got = doc.root().to_value();
    assert_eq!(got, expected);
}

#[test]
fn fixture_small_records_lines_match_serde_json() {
    let path = fixtures_dir().join("twitter_sample_small_records.json");
    let bytes = std::fs::read(&path).unwrap();

    let expected: Vec<serde_json::Value> = bytes
        .split(|&b| b == b'\n')
        .filter(|line| !line.iter().all(u8::is_ascii_whitespace))
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect();

    let doc = doc_from(&bytes, Mode::Lines);
    let got: Vec<serde_json::Value> = doc.lines().map(|n| n.to_value()).collect();

    assert_eq!(got.len(), expected.len());
    for (g, e) in got.iter().zip(expected.iter()) {
        assert_eq!(g, e);
    }
}

// ---------------------------------------------------------------------
// JSON Lines newline-case tests (tape/FORMAT.md §5)
// ---------------------------------------------------------------------

/// (a) A trailing `\n` at EOF is structural (kernel: `bitMapCreatorSimd` in
/// `parse_json_lines.cu` marks every unescaped `\n`), giving one extra
/// tape entry that reads back as a comma right before the artificial
/// close. `Document::lines()` must not surface a phantom trailing value.
#[test]
fn lines_trailing_newline_at_eof() {
    let bytes = b"{\"a\":1}\n";
    let doc = doc_from(bytes, Mode::Lines);
    let got: Vec<serde_json::Value> = doc.lines().map(|n| n.to_value()).collect();
    assert_eq!(got, vec![serde_json::json!({"a": 1})]);
}

/// (b) A blank line between two records is two adjacent `\n` bytes, each
/// independently structural (no skipping) — matches the kernel's
/// unconditional per-`\n` marking. The navigator absorbs the resulting
/// empty span rather than yielding a bogus empty document.
#[test]
fn lines_blank_line_between_records() {
    let bytes = b"{\"a\":1}\n\n{\"a\":2}\n";
    let doc = doc_from(bytes, Mode::Lines);
    let got: Vec<serde_json::Value> = doc.lines().map(|n| n.to_value()).collect();
    assert_eq!(
        got,
        vec![serde_json::json!({"a": 1}), serde_json::json!({"a": 2})]
    );
}

/// (c) CRLF line endings: `\r` matches none of `bitMapCreatorSimd`'s
/// patterns in either mode, so it is never structural and never
/// specially skipped — it is ordinary insignificant whitespace trimmed
/// from the surrounding scalar span like any other whitespace byte.
#[test]
fn lines_crlf_endings() {
    let bytes = b"{\"a\":1}\r\n{\"a\":2}\r\n";
    let doc = doc_from(bytes, Mode::Lines);
    let got: Vec<serde_json::Value> = doc.lines().map(|n| n.to_value()).collect();
    assert_eq!(
        got,
        vec![serde_json::json!({"a": 1}), serde_json::json!({"a": 2})]
    );
}

/// (d) Chunk boundaries are a GPU-side multi-chunk implementation detail
/// (`load_file.cu`'s line-offset chunking always places a chunk boundary
/// immediately after a complete line's `\n`) and have no counterpart in
/// the CPU builder, which never splits its input. Building the same bytes
/// as one input is therefore the correct CPU-side behaviour regardless of
/// where a GPU loader would have cut chunks — assert it is insensitive to
/// where such a cut would fall by checking a cut-independent invariant:
/// the tape built from the whole input matches line-by-line reconstruction.
#[test]
fn lines_chunk_boundary_is_a_noop_for_cpu_builder() {
    let bytes = b"{\"a\":1}\n{\"a\":2}\n{\"a\":3}\n";
    let doc = doc_from(bytes, Mode::Lines);
    let got: Vec<serde_json::Value> = doc.lines().map(|n| n.to_value()).collect();
    let expected: Vec<serde_json::Value> = (1..=3).map(|i| serde_json::json!({"a": i})).collect();
    assert_eq!(got, expected);
}

// ---------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------

fn json_value_strategy() -> impl Strategy<Value = serde_json::Value> {
    let leaf = prop_oneof![
        Just(serde_json::Value::Null),
        any::<bool>().prop_map(serde_json::Value::Bool),
        any::<i32>().prop_map(serde_json::Value::from),
        (-1.0e6f64..1.0e6f64).prop_map(|f| serde_json::json!(f)),
        "[a-zA-Z0-9 _\\-]{0,16}".prop_map(serde_json::Value::String),
    ];
    leaf.prop_recursive(4, 64, 6, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..6).prop_map(serde_json::Value::Array),
            prop::collection::hash_map("[a-zA-Z_][a-zA-Z0-9_]{0,8}", inner, 0..6)
                .prop_map(|m| serde_json::Value::Object(m.into_iter().collect())),
        ]
    })
}

proptest! {
    #[test]
    fn roundtrip_compact(v in json_value_strategy()) {
        let bytes = serde_json::to_vec(&v).unwrap();
        let doc = doc_from(&bytes, Mode::Standard);
        prop_assert_eq!(doc.root().to_value(), v);
    }

    #[test]
    fn roundtrip_pretty(v in json_value_strategy()) {
        let bytes = serde_json::to_vec_pretty(&v).unwrap();
        let doc = doc_from(&bytes, Mode::Standard);
        prop_assert_eq!(doc.root().to_value(), v);
    }

    #[test]
    fn pointer_matches_serde_json(
        v in json_value_strategy(),
        path in prop::collection::vec("[a-zA-Z0-9_]{1,6}", 0..3),
    ) {
        let bytes = serde_json::to_vec(&v).unwrap();
        let doc = doc_from(&bytes, Mode::Standard);

        let pointer = if path.is_empty() {
            String::new()
        } else {
            format!("/{}", path.join("/"))
        };

        let expected = v.pointer(&pointer).cloned();
        let got = doc.pointer(&pointer).map(|n| n.to_value());
        prop_assert_eq!(got, expected);
    }
}

// ---------------------------------------------------------------------
// GPU tapes only define `pair_pos` at openers (FORMAT.md §3); every other
// slot is whatever the pinned allocation held. Nothing may depend on them.

#[test]
fn navigation_and_depth_ignore_non_opener_pair_pos() {
    let path = fixtures_dir().join("twitter_sample_large_record.json");
    let bytes = std::fs::read(path).unwrap();
    let clean = doc_from(&bytes, Mode::Standard);

    let n = clean.tape.len();
    let mut pair_pos: Vec<i32> = clean.tape.pair_pos.to_vec();
    for (idx, slot) in pair_pos.iter_mut().enumerate().take(n - 1).skip(1) {
        let byte = bytes[(clean.tape.structural[idx] - 1) as usize];
        if byte != b'{' && byte != b'[' {
            *slot = 0x5A5A_5A5A_u32 as i32;
        }
    }
    let mut tape = clean.tape.clone();
    tape.pair_pos = pair_pos.into();
    let garbled = Document::new(Cow::Borrowed(&bytes[..]), tape);

    assert_eq!(garbled.depth(), clean.depth());
    assert!(clean.depth() > 1);
    assert_eq!(garbled.root().to_value(), clean.root().to_value());
}

// ---------------------------------------------------------------------
// Standard mode accepts exactly one top-level value. cuJSON only checks
// UTF-8 and bracket balance, so this is enforced by the Document layer.

#[test]
fn standard_mode_accepts_exactly_one_value() {
    use cujson::tape::Error;
    let accepted: &[&[u8]] = &[
        b"{\"a\":1}",
        b" [1, 2] \n",
        b"{}",
        b"42",
        b" \"s\" ",
        b"\"a\\\"b\"",
        b"true",
    ];
    for input in accepted {
        assert!(
            cujson::cpu::parse(input, Mode::Standard).is_ok(),
            "rejected {:?}",
            String::from_utf8_lossy(input)
        );
    }
    let rejected: &[&[u8]] = &[
        b"{\"hello\":\"world\"}\n{\"bonjour\":\"monde\"}",
        b"{\"a\":1} x",
        b"x {\"a\":1}",
        b"[1] [2]",
        b"1 2",
        b"\"a\" \"b\"",
        b"1, 2",
    ];
    for input in rejected {
        assert_eq!(
            cujson::cpu::parse(input, Mode::Standard).err(),
            Some(Error::NotSingleValue),
            "accepted {:?}",
            String::from_utf8_lossy(input)
        );
    }
    // The same JSON Lines input is fine in Lines mode.
    let lines = b"{\"hello\":\"world\"}\n{\"bonjour\":\"monde\"}";
    let doc = cujson::cpu::parse(lines, Mode::Lines).unwrap();
    assert_eq!(doc.lines().count(), 2);
}
