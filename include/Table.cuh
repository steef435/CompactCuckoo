#pragma once

#include "main.h"

// An abstract base class for hash tables to avoid some boilerplate
// (__global__ functions cannot be member functions)
class Table {
public:
    __device__ void clear() {};
    __device__ void insertBatch(const uint64_cu *batch, const size_t batch_len, result *results) {};
    __device__ void warmup() {};
};

__global__ void clear(Table *table) {
    table->clear();
}

__global__ void insertBatch(Table *table, const uint64_cu *batch, const size_t batch_len, result *results) {
    table->insertBatch(batch, batch_len, results);
}

__global__ void warmup(Table *table) {
    table->warmup();
}

// Result helper functions

// Checks if any of the first n results are FAILED
//
// Sets *failed to true if so.
__global__ void hasFailed(bool *failed, const result *results, const size_t n) {
    const auto index = blockIdx.x * blockDim.x + threadIdx.x;
    const auto stride = gridDim.x * blockDim.x;
    for (size_t i = index; i < n; i += stride) {
        if (results[i] == FAILED) *failed = true;
    }
}

// Checks if any of the first n results are FAILED
//
// Returns true if this is the case, false otherwise.
bool hasFailed(const result *results, const size_t n) {
    bool failed, *failed_dev;
    gpuErrchk(cudaMallocManaged(&failed_dev, sizeof(*failed_dev)));
    *failed_dev = false;
    hasFailed<<<n / 512, 512>>>(failed_dev, results, n);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    failed = *failed_dev;
    gpuErrchk(cudaFree(failed_dev));
    return failed;
}