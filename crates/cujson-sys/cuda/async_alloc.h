// Thrust's default temporary-buffer allocator calls cudaMalloc/cudaFree per
// algorithm call, and cudaFree synchronises the whole device, which stops
// parses on other threads from overlapping. This allocator is stream-ordered.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>

struct cujson_async_alloc {
    typedef char value_type;
    char* allocate(std::ptrdiff_t n) {
        void* p = nullptr;
        cudaMallocAsync(&p, static_cast<size_t>(n), cudaStreamPerThread);
        return static_cast<char*>(p);
    }
    void deallocate(char* p, size_t) { cudaFreeAsync(p, cudaStreamPerThread); }
};

// Stateless, so one instance can serve every thread.
static cujson_async_alloc g_cujson_talloc;
