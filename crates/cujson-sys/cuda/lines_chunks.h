// Chunking of a JSON Lines buffer for parse_json_lines. Host-only C++ with no
// CUDA dependency, so tests/lines_chunks.rs compiles it with the host compiler.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace cujson_capi {

struct LinesChunk {
    size_t start;
    size_t size;
};

// Packs whole lines (each ending after a '\n', or at the end of the buffer)
// into consecutive chunks of at most chunk_bytes. A line longer than
// chunk_bytes becomes a chunk on its own. Every chunk is non-empty, because
// parse_json_lines rejects a zero-size chunk by returning an empty result.
inline std::vector<LinesChunk> split_lines_chunks(const uint8_t* data, size_t size, size_t chunk_bytes) {
    // Nothing splits an input that fits in one chunk; skip the pass over it.
    if (size <= chunk_bytes) {
        if (size == 0) return {};
        return {{0, size}};
    }
    std::vector<LinesChunk> chunks;
    size_t chunk_start = 0;
    size_t line_start = 0;
    while (line_start < size) {
        const void* nl = std::memchr(data + line_start, '\n', size - line_start);
        size_t line_end = nl ? static_cast<size_t>(static_cast<const uint8_t*>(nl) - data) + 1 : size;
        if (line_start > chunk_start && line_end - chunk_start > chunk_bytes) {
            chunks.push_back({chunk_start, line_start - chunk_start});
            chunk_start = line_start;
        }
        line_start = line_end;
    }
    if (size > chunk_start) chunks.push_back({chunk_start, size - chunk_start});
    return chunks;
}

} // namespace cujson_capi
