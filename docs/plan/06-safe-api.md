# 06 — Safe `cujson` crate

Depends on: 04 and 05. Verification tier: 1 for the non-`cuda` build; tier 2 for compiling with `cuda`; tier 3 for behaviour.

Join lane A (FFI) and lane B (tape/navigator) into the public API.

## Public API

```rust
pub fn parse(input: &[u8]) -> Result<Document<'_>, Error>;
pub fn parse_owned(input: Vec<u8>) -> Result<Document<'static>, Error>;
pub fn parse_lines(input: &[u8], opts: LinesOptions) -> Result<Document<'_>, Error>;
pub fn parse_file(path: impl AsRef<Path>) -> Result<Document<'static>, Error>;  // reads then parse_owned
pub fn cuda_info() -> Result<CudaInfo, Error>;  // runtime version, devices, compiled archs

#[non_exhaustive]
pub enum Error {
    CudaNotCompiled,              // built without the `cuda` feature
    NoDevice,                     // cujson_device_count() == 0
    InvalidUtf8, Unbalanced,      // from cujson_status
    InputTooLarge { len: usize, max: usize },
    EmptyInput,
    Cuda { code: i32, message: String },
    Internal,
    Io(std::io::Error),
}
```

`LinesOptions { chunk_bytes: usize }`, with a default chosen from upstream's `main_jsonlines_chunksize_MB.cu` default (cite it).

## Implementation

- With `cuda`: call `cujson_sys::cujson_parse_*`, and wrap the returned `cujson_tape` in the tape-storage variant from task 05, whose `Drop` calls `cujson_tape_free`. Structural/pair_pos slices get borrowed from the pinned allocation, not copied
- Check the size limit in Rust before calling C
- A process-wide `static GPU_LOCK: Mutex<()>` covers each parse call. `Document` is `Send + Sync` once built (plain host memory)
- Without `cuda`: every parse function returns `Err(Error::CudaNotCompiled)`. If `cpu-reference` is enabled, **don't** fall back silently. Expose the reference path only as `cujson::cpu::parse` so nobody benchmarks the wrong thing
- `cuda_info()` returns `CudaNotCompiled` without the feature
- Crate docs (`lib.rs` doc comment) cover: what cuJSON is, citation, `cuda` feature, `CUJSON_CUDA_ARCHS`, the driver requirement, the 2 GiB limit, and a short example

## Tests

- Tier 1: without `cuda`, `parse` returns `CudaNotCompiled`
- Tier 3, `#[ignore = "requires GPU"]` in `crates/cujson/tests/gpu.rs`:
  - both fixtures parse and match `serde_json` via `to_value()`
  - GPU tape equals `cpu::parse` tape on the fixtures and on a proptest corpus (fixed seed, a few hundred cases)
  - invalid UTF-8 and unbalanced inputs return errors, and a follow-up valid parse still works (checks the patch 02-2 cleanup)
  - 1000 sequential parses show no device memory growth (`cudaMemGetInfo` through a test-only shim function, or `nvidia-smi` in the runbook)

## Acceptance

- Tier 1 passes; the journal entry lists the ignored GPU tests by name for task 10
