// Pinned host buffers for the tapes handed to callers. cudaMallocHost and
// cudaFreeHost cost ~17 ms and ~7 ms for a 79 MiB tape, so a freed buffer is
// kept (at most one, the largest) for the next parse; cujson_pinned_cache_trim
// releases it.
#pragma once
#include <cstddef>

extern "C" void* cujson_pinned_alloc(size_t bytes);
extern "C" void cujson_pinned_free(void* p);
extern "C" void cujson_pinned_cache_trim(void);

// Plain pinned host allocation for caller input (not the tape cache).
extern "C" void* cujson_host_alloc(size_t bytes);
extern "C" void cujson_host_free(void* p);
