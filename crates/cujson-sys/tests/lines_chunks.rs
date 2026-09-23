//! Tier 1: `split_lines_chunks` (cuda/lines_chunks.h), compiled with the host
//! C++ compiler — no nvcc, no GPU. The chunks it produces are handed to
//! upstream `parse_json_lines`, which rejects a zero-size chunk.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

fn probe() -> &'static PathBuf {
    static PROBE: OnceLock<PathBuf> = OnceLock::new();
    PROBE.get_or_init(|| {
        let cuda_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("cuda");
        let out_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
        let src = out_dir.join("lines_chunks_probe.cpp");
        let bin = out_dir.join("lines_chunks_probe");
        std::fs::write(
            &src,
            r#"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <string>
#include "lines_chunks.h"

int main(int argc, char** argv) {
    size_t chunk_bytes = std::strtoull(argv[1], nullptr, 10);
    std::string data((std::istreambuf_iterator<char>(std::cin)), std::istreambuf_iterator<char>());
    auto chunks = cujson_capi::split_lines_chunks(
        reinterpret_cast<const uint8_t*>(data.data()), data.size(), chunk_bytes);
    for (const auto& c : chunks) std::printf("%zu %zu\n", c.start, c.size);
}
"#,
        )
        .unwrap();
        let status = Command::new(std::env::var("CXX").unwrap_or_else(|_| "c++".into()))
            .args(["-std=c++17", "-O1", "-I"])
            .arg(&cuda_dir)
            .arg(&src)
            .arg("-o")
            .arg(&bin)
            .status()
            .expect("failed to run the host C++ compiler");
        assert!(status.success(), "compiling the lines_chunks probe failed");
        bin
    })
}

fn split(data: &[u8], chunk_bytes: usize) -> Vec<(usize, usize)> {
    let mut child = Command::new(probe())
        .arg(chunk_bytes.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(data).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(|l| {
            let (a, b) = l.split_once(' ').unwrap();
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect()
}

/// Chunks are non-empty, contiguous, cover the buffer, end on a line
/// boundary, and exceed `chunk_bytes` only when they hold a single line.
fn check_invariants(data: &[u8], chunk_bytes: usize) {
    let chunks = split(data, chunk_bytes);
    let ctx = || {
        format!(
            "chunk_bytes={chunk_bytes} data={:?}",
            String::from_utf8_lossy(data)
        )
    };
    let mut pos = 0;
    for &(start, size) in &chunks {
        assert_eq!(start, pos, "gap or overlap: {}", ctx());
        assert!(size > 0, "empty chunk at {start}: {}", ctx());
        let chunk = &data[start..start + size];
        let end = start + size;
        assert!(
            end == data.len() || data[end - 1] == b'\n',
            "chunk splits a line: {}",
            ctx()
        );
        let lines = chunk.iter().filter(|&&b| b == b'\n').count()
            + usize::from(chunk.last() != Some(&b'\n'));
        assert!(
            size <= chunk_bytes || lines == 1,
            "oversized multi-line chunk: {}",
            ctx()
        );
        pos = end;
    }
    assert_eq!(pos, data.len(), "chunks do not cover the buffer: {}", ctx());
}

#[test]
fn first_line_longer_than_chunk_bytes_gets_its_own_chunk() {
    // Regression: the first oversized line used to emit a zero-size chunk.
    let data = b"{\"a\":\"0123456789\"}\n{\"b\":1}\n";
    assert_eq!(split(data, 8), vec![(0, 19), (19, 8)]);
}

#[test]
fn small_lines_pack_up_to_chunk_bytes() {
    let data = b"[1]\n[2]\n[3]\n[4]\n";
    assert_eq!(split(data, 8), vec![(0, 8), (8, 8)]);
    assert_eq!(split(data, 1 << 20), vec![(0, 16)]);
}

#[test]
fn empty_input_has_no_chunks() {
    assert!(split(b"", 16).is_empty());
}

#[test]
fn invariants_hold_on_generated_inputs() {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = |n: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % n
    };
    for _ in 0..300 {
        let mut data = Vec::new();
        for _ in 0..next(12) {
            data.extend(std::iter::repeat_n(b'x', next(40) as usize));
            if next(4) != 0 {
                data.push(b'\n');
            }
        }
        check_invariants(&data, 1 + next(48) as usize);
    }
}
