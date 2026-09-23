// capi_common.cu - version/device info, error strings, tape freeing.
// No upstream parser sources here, so no namespace collision concerns
// with capi_standard.cu/capi_lines.cu (task 02 patch 5).
#include "cujson_capi.h"
#include "pinned_cache.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
std::mutex g_pinned_mu;
std::unordered_map<void*, size_t> g_pinned_cap;  // capacity of every live or cached buffer
std::vector<std::pair<void*, size_t>> g_pinned_cache;  // free buffers kept for reuse
size_t g_pinned_cache_limit = 1;

// Remove and return the smallest cached buffer; caller holds the lock.
std::pair<void*, size_t> pop_smallest_locked() {
    size_t k = 0;
    for (size_t i = 1; i < g_pinned_cache.size(); i++) {
        if (g_pinned_cache[i].second < g_pinned_cache[k].second) k = i;
    }
    auto out = g_pinned_cache[k];
    g_pinned_cache.erase(g_pinned_cache.begin() + k);
    g_pinned_cap.erase(out.first);
    return out;
}
}  // namespace

extern "C" void* cujson_pinned_alloc(size_t bytes) {
    {
        std::lock_guard<std::mutex> lock(g_pinned_mu);
        // Reuse the smallest cached buffer that fits, but only one of similar
        // size, so a small parse does not pin a large one.
        int best = -1;
        for (size_t i = 0; i < g_pinned_cache.size(); i++) {
            size_t cap = g_pinned_cache[i].second;
            if (cap >= bytes && cap <= 2 * bytes &&
                (best < 0 || cap < g_pinned_cache[best].second)) {
                best = static_cast<int>(i);
            }
        }
        if (best >= 0) {
            void* p = g_pinned_cache[best].first;
            g_pinned_cache.erase(g_pinned_cache.begin() + best);
            return p;  // stays in g_pinned_cap with its real capacity
        }
    }
    void* p = nullptr;
    if (cudaMallocHost(&p, bytes) != cudaSuccess) return nullptr;
    std::lock_guard<std::mutex> lock(g_pinned_mu);
    g_pinned_cap[p] = bytes;
    return p;
}

extern "C" void cujson_pinned_free(void* p) {
    if (p == nullptr) return;
    void* release = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_pinned_mu);
        auto it = g_pinned_cap.find(p);
        bool cached = false;
        for (auto& c : g_pinned_cache) cached |= (c.first == p);
        if (cached) return;  // double free
        if (it == g_pinned_cap.end()) {
            release = p;
        } else if (g_pinned_cache.size() < g_pinned_cache_limit) {
            g_pinned_cache.emplace_back(p, it->second);
        } else {
            size_t cap = it->second;
            size_t smallest = g_pinned_cache.empty() ? 0 : g_pinned_cache[0].second;
            for (auto& c : g_pinned_cache) smallest = c.second < smallest ? c.second : smallest;
            if (!g_pinned_cache.empty() && smallest < cap) {
                release = pop_smallest_locked().first;
                g_pinned_cache.emplace_back(p, cap);
            } else {
                g_pinned_cap.erase(it);
                release = p;
            }
        }
    }
    if (release != nullptr) cudaFreeHost(release);
}

extern "C" void cujson_pinned_cache_set_limit(size_t buffers) {
    std::vector<void*> release;
    {
        std::lock_guard<std::mutex> lock(g_pinned_mu);
        g_pinned_cache_limit = buffers < 1 ? 1 : buffers;
        while (g_pinned_cache.size() > g_pinned_cache_limit) {
            release.push_back(pop_smallest_locked().first);
        }
    }
    for (void* p : release) cudaFreeHost(p);
}

extern "C" void cujson_pinned_cache_trim(void) {
    std::vector<void*> release;
    {
        std::lock_guard<std::mutex> lock(g_pinned_mu);
        while (!g_pinned_cache.empty()) release.push_back(pop_smallest_locked().first);
    }
    for (void* p : release) cudaFreeHost(p);
}

extern "C" void* cujson_host_alloc(size_t bytes) {
    void* p = nullptr;
    return cudaMallocHost(&p, bytes) == cudaSuccess ? p : nullptr;
}

extern "C" void cujson_host_free(void* p) {
    if (p != nullptr) cudaFreeHost(p);
}

extern "C" void cujson_tape_free(cujson_tape* tape) {
    if (tape == nullptr) return;
    if (tape->_alloc != nullptr) {
        cujson_pinned_free(tape->_alloc);
    }
    tape->structural = nullptr;
    tape->pair_pos = nullptr;
    tape->len = 0;
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

extern "C" const char* cujson_cuda_error_string(int err) {
    // Fixed-size static buffer: called from behind ffi.rs's process-wide
    // GPU_LOCK, so there is no concurrent writer to race with.
    static char buf[256];
    cudaError_t e = static_cast<cudaError_t>(err);
    std::snprintf(buf, sizeof(buf), "%s: %s", cudaGetErrorName(e), cudaGetErrorString(e));
    return buf;
}

extern "C" int cujson_cuda_driver_version(void) {
    int version = 0;
    cudaError_t err = cudaDriverGetVersion(&version);
    if (err != cudaSuccess) return 0;
    return version;
}

extern "C" cujson_status cujson_mem_get_info(size_t* free_bytes, size_t* total_bytes) {
    if (free_bytes == nullptr || total_bytes == nullptr) return CUJSON_ERR_INTERNAL;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return CUJSON_ERR_CUDA;
    err = cudaMemGetInfo(free_bytes, total_bytes);
    if (err != cudaSuccess) return CUJSON_ERR_CUDA;
    return CUJSON_OK;
}
