// memory_kernels.h
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace gpuburn {

// Sequential bandwidth test - coalesced memory access

void launch_memory_sequential_float(float *data, size_t elements, int cycles,
                                    cudaStream_t stream);
void launch_memory_sequential_double(double *data, size_t elements, int cycles,
                                     cudaStream_t stream);

// Random access pattern - uncached access using device-side cuRAND

void launch_init_rand_states(void *states, size_t n_states, unsigned int seed,
                             cudaStream_t stream);
void launch_memory_random_float(float *data, void *states, size_t elements,
                                int cycles, cudaStream_t stream);
void launch_memory_random_double(double *data, void *states, size_t elements,
                                 int cycles, cudaStream_t stream);

// Stride access pattern - cache thrashing

void launch_memory_stride_float(float *data, size_t elements, size_t stride,
                                int cycles, cudaStream_t stream);
void launch_memory_stride_double(double *data, size_t elements, size_t stride,
                                 int cycles, cudaStream_t stream);

// Validation - compare two buffers

void launch_compare_memory_float(const float *ref, const float *test,
                                 int *faulty, size_t elems,
                                 cudaStream_t stream);
void launch_compare_memory_double(const double *ref, const double *test,
                                  int *faulty, size_t elems,
                                  cudaStream_t stream);
} // namespace gpuburn
