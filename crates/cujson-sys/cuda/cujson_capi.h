/* cujson_capi.h - plain C ABI over cuJSON.
 *
 * The only header Rust's FFI declarations (crates/cujson-sys/src/lib.rs)
 * need to match field-for-field. No C++ type (std::vector, std::string,
 * exceptions) crosses this boundary - every function here is extern "C"
 * and catches every C++ exception internally (see capi_standard.cu,
 * capi_lines.cu, capi_common.cu).
 */
#ifndef CUJSON_CAPI_H
#define CUJSON_CAPI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUJSON_OK = 0,
  CUJSON_ERR_UTF8 = 1,
  CUJSON_ERR_UNBALANCED = 2,
  CUJSON_ERR_INPUT_TOO_LARGE = 3,
  CUJSON_ERR_CUDA = 4,        /* detail in cuda_error */
  CUJSON_ERR_INTERNAL = 5,    /* unexpected C++ exception, or a null out-param */
  CUJSON_ERR_EMPTY_INPUT = 6
} cujson_status;

typedef struct {
  int32_t* structural;   /* length = len, owned, pinned host memory */
  int32_t* pair_pos;     /* length = len, points into the same allocation as structural */
  size_t   len;          /* = cuJSONResult::totalResultSize */
  int32_t  cuda_error;   /* cudaError_t when status == CUJSON_ERR_CUDA, else 0 */
  void*    _alloc;       /* allocation to free; opaque, pass to cujson_tape_free */
} cujson_tape;

/* Parses standard JSON. data/size describe the caller's buffer; cuJSON
 * only reads it (task 02 patch 4). *out is always fully written, even on
 * error (zeroed on failure before CUJSON_OK, populated on success). */
cujson_status cujson_parse_standard(const uint8_t* data, size_t size, cujson_tape* out);

/* Parses JSON Lines. chunk_bytes bounds each chunk's size; a single line
 * longer than chunk_bytes gets a chunk to itself. Chunks are built as
 * pointers into the caller's buffer (data), no extra host allocation or
 * copy for the chunking step itself. */
cujson_status cujson_parse_lines(const uint8_t* data, size_t size, size_t chunk_bytes, cujson_tape* out);

/* Frees a tape's allocation. Safe to call on a zeroed (all-NULL) tape. */
void cujson_tape_free(cujson_tape* tape);

const char* cujson_status_str(cujson_status s);

/* CUDA runtime version (e.g. 12080 for 12.8), or a negated cudaError_t on failure. */
int cujson_cuda_runtime_version(void);

/* Visible CUDA device count, or a negated cudaError_t on failure. */
int cujson_device_count(void);

cujson_status cujson_device_name(int device, char* buf, size_t buf_len);

/* e.g. "75,80,86,89,90;ptx90" - set by build.rs via -DCUJSON_COMPILED_ARCHS. */
const char* cujson_compiled_archs(void);

/* "<cudaGetErrorName>: <cudaGetErrorString>" for a raw cudaError_t, e.g.
 * "cudaErrorInsufficientDriver: CUDA driver version is insufficient for
 * CUDA runtime version". Returned pointer is to a static buffer, valid
 * until the next call on this thread; callers must copy it out before
 * calling again. */
const char* cujson_cuda_error_string(int err);

/* Driver's supported CUDA version (e.g. 12020 for 12.2), or 0 if no
 * driver responds (cudaDriverGetVersion failed). */
int cujson_cuda_driver_version(void);

/* cudaDeviceSynchronize then cudaMemGetInfo, both in bytes. */
cujson_status cujson_mem_get_info(size_t* free_bytes, size_t* total_bytes);

#ifdef __cplusplus
}
#endif

#endif /* CUJSON_CAPI_H */
