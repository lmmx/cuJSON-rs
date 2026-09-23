# 03 — C ABI shim over cuJSON

Depends on: 02. Lane A. Verification tier: none here for the `.cu` side; tier 2 via task 09.

The shim is the only code Rust links against. It hides C++ types (`std::vector`, exceptions, `std::string`) behind a plain C interface.

## Files

- `crates/cujson-sys/cuda/cujson_capi.h`: the C header, shared verbatim with the Rust FFI declarations in task 04
- `crates/cujson-sys/cuda/capi_standard.cu`: `#include`s the patched standard-JSON upstream sources (unity TU) and defines the standard-JSON entry points
- `crates/cujson-sys/cuda/capi_lines.cu`: the same for JSON Lines
- `crates/cujson-sys/cuda/capi_common.cu`: version, device info, error strings, and result freeing

These are separate TUs because of patch 02-5. Each gets compiled by cudaforge as its own object into one static archive.

## API (C)

```c
typedef enum {
  CUJSON_OK = 0,
  CUJSON_ERR_UTF8 = 1,
  CUJSON_ERR_UNBALANCED = 2,
  CUJSON_ERR_INPUT_TOO_LARGE = 3,
  CUJSON_ERR_CUDA = 4,        /* detail in cuda_error */
  CUJSON_ERR_INTERNAL = 5,    /* unexpected C++ exception */
  CUJSON_ERR_EMPTY_INPUT = 6
} cujson_status;

typedef struct {
  int32_t* structural;   /* length = len, owned, pinned host memory */
  int32_t* pair_pos;     /* length = len, points into the same allocation */
  size_t   len;          /* = cuJSONResult::totalResultSize */
  int32_t  depth;
  int32_t  cuda_error;   /* cudaError_t when status == CUJSON_ERR_CUDA, else 0 */
  void*    _alloc;       /* allocation to free; opaque */
} cujson_tape;

cujson_status cujson_parse_standard(const uint8_t* data, size_t size, cujson_tape* out);
cujson_status cujson_parse_lines(const uint8_t* data, size_t size, size_t chunk_bytes, cujson_tape* out);
void          cujson_tape_free(cujson_tape* tape);             /* safe on zeroed tape */
const char*   cujson_status_str(cujson_status s);
int           cujson_cuda_runtime_version(void);
int           cujson_device_count(void);                        /* <0 = cudaError, negated */
cujson_status cujson_device_name(int device, char* buf, size_t buf_len);
const char*   cujson_compiled_archs(void);                     /* e.g. "75,80,86,89,90;ptx90", from a -D define set by build.rs */
```

Treat the exact field set as a proposal. Change it if the upstream result layout needs something else, and document the change in the journal entry. Task 05 needs `structural`, `pair_pos`, `len` and the JSON Lines chunk metadata; check what `cuJSONResult::resultSizes`/`resultSizesPrefix` contribute for JSON Lines and expose them if the navigator needs them.

## Behaviour

- Every exported function is `extern "C"` and wraps its body in `try { … } catch (const cujson_error& e) { … } catch (...) { return CUJSON_ERR_INTERNAL; }`, so no exception crosses the ABI boundary
- After a parse, call `cudaGetLastError()` and `cudaDeviceSynchronize()`; if either returns an error, free the result and return `CUJSON_ERR_CUDA`
- Reject `size == 0` (`EMPTY_INPUT`) and sizes at or above `INT32_MAX` minus upstream's padding (`INPUT_TOO_LARGE`) before any allocation
- Accept pageable input. Upstream's `cudaMemcpy` handles pageable memory; pinned-input fast paths are out of scope
- `cujson_parse_lines` builds the `cuJSONLinesInput` chunk vectors itself from `data`, splitting at `\n` boundaries into chunks of at most `chunk_bytes` (one line longer than `chunk_bytes` gets a chunk to itself), mirroring `loadJSONLines_chunkSizeBytes` in `load_file.cu` but without file I/O or a second allocation
- File loading (`load_file.cu`) doesn't go into the shim; Rust reads files

## Acceptance

- Header compiles as C with `cc -fsyntax-only -x c cujson_capi.h` (tier 1: the header has no CUDA dependency)
- Journal entry marks the `.cu` side as unverified until tier 2
