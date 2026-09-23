// capi_standard.cu - C ABI entry point for standard JSON.
// Unity TU: pulls in the patched standard-JSON upstream sources (namespace
// cujson_std, task 02 patch 5) plus the query iterator, all under one
// translation unit so their file-scope/static symbols don't need to be
// visible anywhere else.
#include "cujson_capi.h"
#include "upstream/cujson.h"
#include "upstream/cujson_error.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <climits>

extern "C" cujson_status cujson_parse_standard(const uint8_t* data, size_t size, cujson_tape* out) {
    if (out == nullptr) return CUJSON_ERR_INTERNAL;
    out->structural = nullptr;
    out->pair_pos = nullptr;
    out->len = 0;
    out->depth = 0;
    out->cuda_error = 0;
    out->_alloc = nullptr;

    if (data == nullptr || size == 0) return CUJSON_ERR_EMPTY_INPUT;
    // parse_standard_json pads input.size up to 3 bytes to a 4-byte boundary
    // and works in `int`-sized token counts; stay comfortably clear of
    // INT32_MAX so that padding and any token-count arithmetic can't wrap.
    if (size >= static_cast<size_t>(INT32_MAX) - 8) return CUJSON_ERR_INPUT_TOO_LARGE;

    cuJSONInput input;
    input.data = const_cast<uint8_t*>(data);
    input.size = size;

    cuJSONResult result{};
    try {
        result = cujson_std::parse_standard_json(input);
    } catch (const cujson_error& e) {
        switch (e.code) {
            case CUJSON_ERR_UTF8:
            case CUJSON_ERR_UNBALANCED:
                return static_cast<cujson_status>(e.code);
            default:
                return CUJSON_ERR_INTERNAL;
        }
    } catch (...) {
        return CUJSON_ERR_INTERNAL;
    }

    cudaError_t cerr = cudaGetLastError();
    if (cerr == cudaSuccess) cerr = cudaDeviceSynchronize();
    if (cerr != cudaSuccess) {
        if (result.structural != nullptr) cudaFreeHost(result.structural);
        out->cuda_error = static_cast<int32_t>(cerr);
        return CUJSON_ERR_CUDA;
    }

    if (result.structural == nullptr) {
        // parse_standard_json's own input.data==nullptr/size==0 guard returns
        // cuJSONResult{} without throwing; unreachable given the checks above,
        // kept as a defensive fallback.
        return CUJSON_ERR_INTERNAL;
    }

    out->structural = result.structural;
    out->pair_pos = result.pair_pos;
    out->len = static_cast<size_t>(result.totalResultSize);
    out->depth = static_cast<int32_t>(result.depth);
    out->_alloc = result.structural;
    return CUJSON_OK;
}
