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
