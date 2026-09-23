#include <stdio.h>
#include <iostream>
#include <stdint.h>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <string.h>
#include <bitset>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/extrema.h>
#include <thrust/partition.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <inttypes.h>
#include <thrust/host_vector.h>
#include <device_launch_parameters.h>
#include <vector>
#include <cstring>
#include <cub/cub.cuh>




// #include "./12-GJSON-Class.cuh"

#define         MAXLINELENGTH     1073741824   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824// Max record size
                                              //4MB       8MB     32BM    64MB      128MB    192MB     256MB     512MB     768MB       1GB
#define         BUFSIZE           1073741824   //4194304 8388608 33554432 67108864 134217728 201326592 268435456 536870912 805306368 1073741824

#define BLOCKSIZE 256

#define OPENBRACKET 91
#define CLOSEBRACKET 93
#define OPENBRACE 123
#define CLOSEBRACE 125
#define I 73

#define ROW1 1
#define ROW2 2
#define ROW3 3
#define ROW4 4
#define ROW5 5

#ifndef DEBUG_MODE
    #define DEBUG_MODE 1 
    // Set to 5 for debugging (memory consumption),
    // Set to 4 for debugging (size),
    // Set to 3 for debugging (overall time report), 
    // Set to 2 for debugging (time report), 
    // Set to 1 for debugging (print), 0 for production
#endif

using namespace std;
using namespace std::chrono;

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip> // For formatting




// Struct to check if a given integer is equal to 1.
struct is_one {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    bool operator()(const int x) { 
        return (x == 1); // Returns true if x is 1.
    }
};

// Struct to check if a character is an opening brace or bracket.
struct is_opening {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    bool operator()(char x) {
        return (x == OPENBRACE) || (x == OPENBRACKET); // Returns true for '{' or '['.
    }
};

// Struct to check if a character is a closing brace or bracket.
struct is_closing {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    bool operator()(char x) {
        return (x == CLOSEBRACE) || (x == CLOSEBRACKET); // Returns true for '}' or ']'.
    }
};

// Struct to decrease an integer by 1.
struct decrease {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    int operator()(int x) {
        return x - 1; // Decreases the input integer by 1.
    }
};

// Struct to increase an integer by 1.
struct increase {
    __host__ __device__ // Can be called from both host (CPU) and device (GPU) code.
    int operator()(int x) {
        return x + 1; // Increases the input integer by 1.
    }
};





// Inline device function to compute the prefix XOR for a 32-bit integer.
// This function performs XOR-based prefix computations for efficiency.
__device__ __forceinline__
uint32_t prefix_xor(uint32_t x) {
    x ^= (x << 1);   // XOR with left-shifted version by 1 bit.
    x ^= (x << 2);   // XOR with left-shifted version by 2 bits.
    x ^= (x << 4);   // XOR with left-shifted version by 4 bits.
    x ^= (x << 8);   // XOR with left-shifted version by 8 bits.
    x ^= (x << 16);  // XOR with left-shifted version by 16 bits.
    return x;        // Returns the resulting XOR value.
}

// Inline device function to compute the prefix XOR for a 64-bit integer.
// This function performs XOR-based prefix computations for efficiency.
__device__ __forceinline__
uint64_t prefix_xor64(uint64_t x) {
    x ^= (x << 1);   // XOR with left-shifted version by 1 bit.
    x ^= (x << 2);   // XOR with left-shifted version by 2 bits.
    x ^= (x << 4);   // XOR with left-shifted version by 4 bits.
    x ^= (x << 8);   // XOR with left-shifted version by 8 bits.
    x ^= (x << 16);  // XOR with left-shifted version by 16 bits.
    x ^= (x << 32);  // XOR with left-shifted version by 32 bits.
    return x;        // Returns the resulting XOR value.
}





// ______________________check_CUDA_______________________
// Function to check the status of a CUDA API call and handle errors if any.
// If the CUDA call fails, the function prints the error message and terminates the program.
void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) { // Check if the CUDA call did not succeed.
        // Print the error message associated with the CUDA error.
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        // Exit the program with a non-zero status to indicate an error.
        exit(1);
    }
}

// CUB functions:

// count_ones_cub: Counts the number of ones in a device array using CUB.
// This function uses the CUB library to perform a parallel reduction on the device array.
// It first queries the temporary storage needed for the reduction, then allocates that storage,
// performs the reduction, and finally copies the result back to the host.
// The function returns the count of ones found in the device array.
// The input array is expected to be a device pointer to an array of uint8_t values.
// The length parameter specifies the number of elements in the array.
// The function returns the count of ones found in the device array.
uint32_t count_ones_cub(uint8_t* d_flags, size_t length){
    // 1. temporary‑storage query
    void*  d_temp  = nullptr;
    size_t temp_sz = 0;
    uint32_t* d_result;                      // device scalar result
    cudaMalloc(&d_result, sizeof(uint32_t));

    cub::DeviceReduce::Sum(
        d_temp,     temp_sz,                 // temp buffer ptr & size
        d_flags,    d_result,                // in, out
        length);                             // # elements

    // 2. allocate temp storage and run the real reduction
    cudaMalloc(&d_temp, temp_sz);

    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        d_flags, d_result,
        length);

    // 3. copy result back to host
    uint32_t h_count = 0;
    cudaMemcpy(&h_count, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 4. clean‑up
    cudaFree(d_temp);
    cudaFree(d_result);

    return h_count;
}

uint32_t reduce_cub_int(int8_t* d_flags, size_t length){
    // 1. temporary‑storage query
    void*  d_temp  = nullptr;
    size_t temp_sz = 0;
    uint32_t* d_result;                      // device scalar result
    cudaMalloc(&d_result, sizeof(uint32_t));

    cub::DeviceReduce::Sum(
        d_temp,     temp_sz,                 // temp buffer ptr & size
        d_flags,    d_result,                // in, out
        length);                             // # elements

    // 2. allocate temp storage and run the real reduction
    cudaMalloc(&d_temp, temp_sz);

    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        d_flags, d_result,
        length);

    // 3. copy result back to host
    uint32_t h_count = 0;
    cudaMemcpy(&h_count, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // 4. clean‑up
    cudaFree(d_temp);
    cudaFree(d_result);

    return h_count;
}



// inclusive_scan_inplace_cub: Performs an inclusive scan on a device array using CUB.
// This function uses the CUB library to perform an inclusive scan (prefix sum) on the input array.
// The input array is expected to be a device pointer to an array of int8_t values.
// The length parameter specifies the number of elements in the array.
// The function modifies the input array in place, storing the result back in the same array.
// The function does not return any value.
// It allocates temporary storage for the scan operation, performs the scan, and then frees the temporary storage.
// The input array is expected to be a device pointer to an array of int8_t values.
// The length parameter specifies the number of elements in the array.
// The function modifies the input array in place, storing the result back in the same array.
// The function does not return any value.
// It allocates temporary storage for the scan operation, performs the scan, and then frees the temporary storage.
// The function is designed to be efficient and uses CUB's parallel algorithms for the scan operation.
void inclusive_scan_inplace_cub(int8_t* d_data, size_t length) {
    // Allocate temp buffer
    void* d_temp = nullptr;
    size_t temp_bytes = 0;

    // First call: query temp storage
    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        d_data, d_data,  // in-place
        length
    );

    // Allocate temp buffer
    cudaMalloc(&d_temp, temp_bytes);

    // Second call: actual inclusive scan
    cub::DeviceScan::InclusiveSum(
        d_temp, temp_bytes,
        d_data, d_data,
        length
    );

    // Free temp storage
    cudaFree(d_temp);
}



// A small helper to count “1”s in a virtual flag stream using CUB
template<class FlagIter>
uint32_t count_virtual_flags_cub(FlagIter flags, size_t length ) {
    // device storage for the result
    uint32_t* d_result = nullptr;
    cudaMalloc(&d_result, sizeof(uint32_t));

    // 1a) query temp storage size
    void*   d_temp   = nullptr;
    size_t  temp_sz  = 0;
    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        flags,    // could be a transform_iterator
        d_result, // device scalar
        length
    );

    // 1b) allocate temp storage & run real reduction
    cudaMalloc(&d_temp, temp_sz);
    cub::DeviceReduce::Sum(
        d_temp, temp_sz,
        flags,
        d_result,
        length
    );

    // 1c) copy back
    uint32_t h_result = 0;
    cudaMemcpy(&h_result, d_result, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // clean up
    cudaFree(d_temp);
    cudaFree(d_result);

    return h_result;
}


// Copies elements from d_token_indices to d_selected_token_indices where d_output_flag == 1
// This function uses CUB's DeviceSelect::Flagged to perform the scatter operation.
// The input array d_token_indices is expected to be a device pointer to an array of uint32_t values.
// The d_output_flag array is a device pointer to an array of uint8_t values, where 1 indicates the element should be copied.
// The d_selected_token_indices array is a device pointer to an array of uint32_t values, where the selected elements will be copied.
// The tokens_count parameter specifies the number of elements in the input array.
// The function does not return any value.
// It allocates temporary storage for the scatter operation, performs the scatter, and then frees the temporary storage.
// The function is designed to be efficient and uses CUB's parallel algorithms for the scatter operation.

void scatter_cub(
    const uint32_t* d_token_indices,        // input data
    const uint8_t* d_output_flag,           // 0/1 flag for selection
    uint32_t* d_selected_token_indices,     // output buffer (preallocated)
    size_t tokens_count                     // number of input items
) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    uint32_t* d_num_selected_out;

    // Allocate temporary output count
    cudaMalloc(&d_num_selected_out, sizeof(uint32_t));

    // Step 1: Query temporary storage size
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        d_token_indices,            // input values
        d_output_flag,              // stencil
        d_selected_token_indices,   // output
        d_num_selected_out,         // number selected
        tokens_count
    );

    // Step 2: Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Step 3: Run the actual selection
    cub::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes,
        d_token_indices,
        d_output_flag,
        d_selected_token_indices,
        d_num_selected_out,
        tokens_count
    );

    // Optional: get count back (if you want to use it later)
    // uint32_t h_selected_count = 0;
    // cudaMemcpy(&h_selected_count, d_num_selected_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_temp_storage);
    cudaFree(d_num_selected_out);
}


template<class FlagIter>
void scatter_virtual_flag_cub(
    const uint32_t*        d_token_indices,         // input indices
    size_t                 tokens_count,           // number of tokens
    FlagIter               flags,                  // transform_iterator over [0..tokens_count)
    uint32_t*              d_selected_tokens,      // OUT: compacted indices
    uint32_t*              d_selected_count       // OUT: device scalar count
) {
    // 2a) query temp storage
    void*  d_temp  = nullptr;
    size_t temp_sz = 0;
    auto  index_begin = thrust::make_counting_iterator<uint32_t>(0);

    cub::DeviceSelect::Flagged(
       d_temp, temp_sz,
       index_begin,     // the “input items” (we really only care about the index)
       flags,           // the lazy flag stream
       d_selected_tokens,
       d_selected_count,
       tokens_count
    );

    // 2b) allocate & run
    cudaMalloc(&d_temp, temp_sz);
    cub::DeviceSelect::Flagged(
       d_temp, temp_sz,
       index_begin,
       flags,
       d_selected_tokens,
       d_selected_count,
       tokens_count
    );
    cudaFree(d_temp);
}



