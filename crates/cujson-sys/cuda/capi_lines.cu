// capi_lines.cu - C ABI entry point for JSON Lines.
// Unity TU: pulls in the patched JSON-Lines upstream sources (namespace
// cujson_lines, task 02 patch 5). Kept separate from capi_standard.cu so
// each becomes its own object in the static archive (task 02 patch 5's
// namespacing is what lets both objects link into one binary).
#include "cujson_capi.h"
#include "upstream/cujsonlines.h"
#include "upstream/cujson_error.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <climits>
#include <vector>

namespace {

// Mirrors loadJSONLines_chunkSizeBytes (upstream/load_file.cu) but reads
// from the caller's buffer directly instead of a file, and its chunks
// point into that buffer instead of a second pinned-memory copy.
void build_lines_chunks(const uint8_t* data, size_t size, size_t chunk_bytes, cuJSONLinesInput& input) {
    input.data = const_cast<uint8_t*>(data);
    input.size = size;

    std::vector<size_t> line_offsets;
    line_offsets.push_back(0);
    for (size_t i = 0; i < size; ++i) {
        if (data[i] == '\n') line_offsets.push_back(i + 1);
    }
    if (line_offsets.back() < size) line_offsets.push_back(size);

    size_t current_chunk_start = 0;
    size_t current_offset = 0;
    size_t chunk_count = 0;
    for (size_t i = 1; i < line_offsets.size(); ++i) {
        size_t line_start = line_offsets[i - 1];
        size_t line_end = line_offsets[i];

        // A line longer than chunk_bytes still gets a chunk to itself: the
        // finalize-and-restart below always uses the line's own start, so a
        // single oversized line just produces an oversized chunk here and
        // starts fresh on the next line.
        if ((line_end - current_chunk_start) > chunk_bytes) {
            chunk_count++;
            input.chunks.push_back(const_cast<uint8_t*>(data) + current_chunk_start);
            input.chunksSize.push_back(current_offset - current_chunk_start);
            current_chunk_start = line_start;
        }

        current_offset = line_end;
    }

    if (current_chunk_start < size) {
        input.chunks.push_back(const_cast<uint8_t*>(data) + current_chunk_start);
        input.chunksSize.push_back(size - current_chunk_start);
        chunk_count++;
    }

    input.chunkCount = chunk_count;
}

} // namespace

extern "C" cujson_status cujson_parse_lines(const uint8_t* data, size_t size, size_t chunk_bytes, cujson_tape* out) {
    if (out == nullptr) return CUJSON_ERR_INTERNAL;
    out->structural = nullptr;
    out->pair_pos = nullptr;
    out->len = 0;
    out->depth = 0;
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
        // parse_json_lines's own guard clauses return cuJSONResult{} without
        // throwing; unreachable given the checks above and build_lines_chunks
        // always producing well-formed chunk metadata, kept as a fallback.
        return CUJSON_ERR_INTERNAL;
    }

    out->structural = result.structural;
    out->pair_pos = result.pair_pos;
    out->len = static_cast<size_t>(result.totalResultSize);
    out->depth = static_cast<int32_t>(result.depth);
    out->_alloc = result.structural;
    return CUJSON_OK;
}
