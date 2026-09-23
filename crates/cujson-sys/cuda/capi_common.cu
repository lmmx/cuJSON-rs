// capi_common.cu - version/device info, error strings, tape freeing.
// No upstream parser sources here, so no namespace collision concerns
// with capi_standard.cu/capi_lines.cu (task 02 patch 5).
#include "cujson_capi.h"

#include <cuda_runtime.h>
#include <cstdio>

extern "C" void cujson_tape_free(cujson_tape* tape) {
    if (tape == nullptr) return;
    if (tape->_alloc != nullptr) {
        cudaFreeHost(tape->_alloc);
    }
    tape->structural = nullptr;
    tape->pair_pos = nullptr;
    tape->len = 0;
    tape->depth = 0;
    tape->cuda_error = 0;
    tape->_alloc = nullptr;
}

extern "C" const char* cujson_status_str(cujson_status s) {
    switch (s) {
        case CUJSON_OK: return "ok";
        case CUJSON_ERR_UTF8: return "invalid UTF-8 in input";
        case CUJSON_ERR_UNBALANCED: return "unbalanced JSON structure";
        case CUJSON_ERR_INPUT_TOO_LARGE: return "input too large";
        case CUJSON_ERR_CUDA: return "CUDA runtime error";
        case CUJSON_ERR_INTERNAL: return "internal error";
        case CUJSON_ERR_EMPTY_INPUT: return "empty input";
    }
    return "unknown cujson_status";
}

extern "C" int cujson_cuda_runtime_version(void) {
    int version = 0;
    cudaError_t err = cudaRuntimeGetVersion(&version);
    if (err != cudaSuccess) return -static_cast<int>(err);
    return version;
}

extern "C" int cujson_device_count(void) {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return -static_cast<int>(err);
    return count;
}

extern "C" cujson_status cujson_device_name(int device, char* buf, size_t buf_len) {
    if (buf == nullptr || buf_len == 0) return CUJSON_ERR_INTERNAL;
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return CUJSON_ERR_CUDA;
    std::snprintf(buf, buf_len, "%s", prop.name);
    return CUJSON_OK;
}

#ifndef CUJSON_COMPILED_ARCHS
#define CUJSON_COMPILED_ARCHS "unknown"
#endif

extern "C" const char* cujson_compiled_archs(void) {
    return CUJSON_COMPILED_ARCHS;
}
