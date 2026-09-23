//! Compiled with `cuda`, on a host where CUDA is unusable (no driver or
//! device); each test returns early when a GPU is usable. Confirms `parse`/`parse_lines` report the CUDA failure as
//! `Error::Cuda`/`Error::NoDevice` — never `Error::Internal` — matching
//! `cuda_info()`'s error path. Before the fix (task 07 item 6), cuJSON's
//! C++ threw `thrust::system_error` on an unusable device and a generic
//! `catch (...)` in `capi_standard.cu`/`capi_lines.cu` mapped it to
//! `CUJSON_ERR_INTERNAL`, losing the underlying `cudaError_t`.
#![cfg(feature = "cuda")]

fn gpu_usable() -> bool {
    let usable = cujson::cuda_info().is_ok();
    if usable {
        eprintln!("skipped: a usable GPU is present");
    }
    usable
}

fn assert_cuda_or_no_device(err: cujson::Error, label: &str) {
    match err {
        cujson::Error::Cuda { code, .. } => {
            assert!(
                code > 0,
                "{label}: Cuda error with non-positive code {code}"
            );
        }
        cujson::Error::NoDevice => {}
        other => panic!("{label}: expected Error::Cuda or Error::NoDevice, got {other:?}"),
    }
}

#[test]
fn parse_standard_reports_cuda_error_not_internal() {
    if gpu_usable() {
        return;
    }
    let err = cujson::parse(br#"{"a":1}"#).unwrap_err();
    assert_cuda_or_no_device(err, "parse");
}

#[test]
fn parse_lines_reports_cuda_error_not_internal() {
    if gpu_usable() {
        return;
    }
    let err = cujson::parse_lines(b"{\"a\":1}\n", cujson::LinesOptions::default()).unwrap_err();
    assert_cuda_or_no_device(err, "parse_lines");
}
