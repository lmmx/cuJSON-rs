//! Tier 1: checks that the hand-written `#[repr(C)] cujson_tape` in
//! src/lib.rs has the same size/align/field offsets, and that the
//! hand-written `CUJSON_*` constants have the same values, as a real C
//! compile of cujson_capi.h. Uses the host `cc` (not nvcc), so this runs
//! without the `cuda` feature and without a CUDA toolkit.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn cujson_tape_matches_c_header() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let cuda_dir = manifest_dir.join("cuda");
    let header = cuda_dir.join("cujson_capi.h");
    assert!(header.is_file(), "missing {}", header.display());

    let out_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    let src = out_dir.join("layout_probe.c");
    let bin = out_dir.join("layout_probe");

    std::fs::write(
        &src,
        r#"
#include <stdio.h>
#include <stddef.h>
#include "cujson_capi.h"

int main(void) {
    printf("%zu %zu %zu %zu %zu %zu %zu %zu %d %d %d %d %d %d %d\n",
        sizeof(cujson_tape),
        (size_t)_Alignof(cujson_tape),
        offsetof(cujson_tape, structural),
        offsetof(cujson_tape, pair_pos),
        offsetof(cujson_tape, len),
        offsetof(cujson_tape, depth),
        offsetof(cujson_tape, cuda_error),
        offsetof(cujson_tape, _alloc),
        (int)CUJSON_OK,
        (int)CUJSON_ERR_UTF8,
        (int)CUJSON_ERR_UNBALANCED,
        (int)CUJSON_ERR_INPUT_TOO_LARGE,
        (int)CUJSON_ERR_CUDA,
        (int)CUJSON_ERR_INTERNAL,
        (int)CUJSON_ERR_EMPTY_INPUT);
    return 0;
}
"#,
    )
    .expect("write scratch C source");

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let status = Command::new(&cc)
        .arg("-std=c11")
        .arg("-I")
        .arg(&cuda_dir)
        .arg(&src)
        .arg("-o")
        .arg(&bin)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn {cc}: {e}"));
    assert!(status.success(), "{cc} failed to compile the layout probe");

    let output = Command::new(&bin)
        .output()
        .expect("failed to run layout probe");
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).expect("probe output not UTF-8");
    let fields: Vec<usize> = stdout
        .split_whitespace()
        .take(8)
        .map(|s| s.parse().expect("probe printed a non-numeric field"))
        .collect();
    let statuses: Vec<i32> = stdout
        .split_whitespace()
        .skip(8)
        .map(|s| s.parse().expect("probe printed a non-numeric status"))
        .collect();

    assert_eq!(
        fields[0],
        core::mem::size_of::<cujson_sys::cujson_tape>(),
        "size_of"
    );
    assert_eq!(
        fields[1],
        core::mem::align_of::<cujson_sys::cujson_tape>(),
        "align_of"
    );
    assert_eq!(
        fields[2],
        core::mem::offset_of!(cujson_sys::cujson_tape, structural),
        "offsetof(structural)"
    );
    assert_eq!(
        fields[3],
        core::mem::offset_of!(cujson_sys::cujson_tape, pair_pos),
        "offsetof(pair_pos)"
    );
    assert_eq!(
        fields[4],
        core::mem::offset_of!(cujson_sys::cujson_tape, len),
        "offsetof(len)"
    );
    assert_eq!(
        fields[5],
        core::mem::offset_of!(cujson_sys::cujson_tape, depth),
        "offsetof(depth)"
    );
    assert_eq!(
        fields[6],
        core::mem::offset_of!(cujson_sys::cujson_tape, cuda_error),
        "offsetof(cuda_error)"
    );
    assert_eq!(
        fields[7],
        core::mem::offset_of!(cujson_sys::cujson_tape, _alloc),
        "offsetof(_alloc)"
    );

    assert_eq!(statuses[0], cujson_sys::CUJSON_OK);
    assert_eq!(statuses[1], cujson_sys::CUJSON_ERR_UTF8);
    assert_eq!(statuses[2], cujson_sys::CUJSON_ERR_UNBALANCED);
    assert_eq!(statuses[3], cujson_sys::CUJSON_ERR_INPUT_TOO_LARGE);
    assert_eq!(statuses[4], cujson_sys::CUJSON_ERR_CUDA);
    assert_eq!(statuses[5], cujson_sys::CUJSON_ERR_INTERNAL);
    assert_eq!(statuses[6], cujson_sys::CUJSON_ERR_EMPTY_INPUT);
}
