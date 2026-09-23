// capi_lines.cu - C ABI entry point for JSON Lines.
// Unity TU: pulls in the patched JSON-Lines upstream sources (namespace
// cujson_lines, task 02 patch 5). Kept separate from capi_standard.cu so
// each becomes its own object in the static archive (task 02 patch 5's
// namespacing is what lets both objects link into one binary).
#include "cujson_capi.h"
#include "upstream/cujsonlines.h"
#include "upstream/cujson_error.h"
#include "lines_chunks.h"

#include <cuda_runtime.h>
#include "pinned_cache.h"
#include <thrust/system_error.h>
#include <cstdint>
#include <climits>
#include <vector>

namespace {

void build_lines_chunks(const uint8_t* data, size_t size, size_t chunk_bytes, cuJSONLinesInput& input) {
    input.data = const_cast<uint8_t*>(data);
    input.size = size;
    for (const cujson_capi::LinesChunk& c : cujson_capi::split_lines_chunks(data, size, chunk_bytes)) {
        input.chunks.push_back(const_cast<uint8_t*>(data) + c.start);
        input.chunksSize.push_back(c.size);
    }
    input.chunkCount = input.chunks.size();
}

} // namespace

extern "C" cujson_status cujson_parse_lines(const uint8_t* data, size_t size, size_t chunk_bytes, cujson_tape* out) {
    if (out == nullptr) return CUJSON_ERR_INTERNAL;
    out->structural = nullptr;
    out->pair_pos = nullptr;
    out->len = 0;
    out->cuda_error = 0;
    out->_alloc = nullptr;

    if (data == nullptr || size == 0) return CUJSON_ERR_EMPTY_INPUT;
    if (size >= static_cast<size_t>(INT32_MAX) - 8) return CUJSON_ERR_INPUT_TOO_LARGE;
    if (chunk_bytes == 0) chunk_bytes = size; // whole input as a single chunk

    cuJSONLinesInput input;
    build_lines_chunks(data, size, chunk_bytes, input);
    if (input.chunkCount == 0) return CUJSON_ERR_EMPTY_INPUT;

    cuJSONResult result{};
    try {
        result = cujson_lines::parse_json_lines(input);
    } catch (const cujson_error& e) {
        switch (e.code) {
            case CUJSON_ERR_UTF8:
            case CUJSON_ERR_UNBALANCED:
                return static_cast<cujson_status>(e.code);
            default:
                return CUJSON_ERR_INTERNAL;
        }
    } catch (const thrust::system_error& e) {
        // See capi_standard.cu's identical catch clause.
        out->cuda_error = static_cast<int32_t>(e.code().value());
        return CUJSON_ERR_CUDA;
    } catch (...) {
        cudaError_t cerr = cudaGetLastError();
        if (cerr != cudaSuccess) {
            out->cuda_error = static_cast<int32_t>(cerr);
            return CUJSON_ERR_CUDA;
        }
        return CUJSON_ERR_INTERNAL;
    }

    cudaError_t cerr = cudaGetLastError();
    if (cerr == cudaSuccess) cerr = cudaStreamSynchronize(0);  // this thread's stream, so parses on other threads keep running
    if (cerr != cudaSuccess) {
        if (result.structural != nullptr) cujson_pinned_free(result.structural);
        out->cuda_error = static_cast<int32_t>(cerr);
        return CUJSON_ERR_CUDA;
    }

    if (result.structural == nullptr) {
        // parse_json_lines's guard clauses return cuJSONResult{} without
        // throwing, e.g. for a zero-size chunk; split_lines_chunks never
        // produces one.
        return CUJSON_ERR_INTERNAL;
    }

    // mergeChunks leaves both artificial wrapper entries unwritten in its
    // uninitialised pinned buffer. Write them as parse_standard_json does
    // (FORMAT.md §2), so both modes return a fully defined tape. The last
    // entry is also pair_pos[0].
    const size_t n = static_cast<size_t>(result.totalResultSize);
    result.structural[0] = 0;
    result.structural[n - 1] = static_cast<int32_t>(n - 1);

    out->structural = result.structural;
    out->pair_pos = result.pair_pos;
    out->len = static_cast<size_t>(result.totalResultSize);
    out->_alloc = result.structural;
    return CUJSON_OK;
}
