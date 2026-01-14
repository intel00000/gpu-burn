// compute_kernels.h
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace gpuburn {

// Init buffer with deterministic values (both halves identical)

void launch_compute_verify_reinit_float(float *data, int *faulty,
                                        size_t elements, uint32_t seed,
                                        cudaStream_t stream);
void launch_compute_verify_reinit_double(double *data, int *faulty,
                                         size_t elements, uint32_t seed,
                                         cudaStream_t stream);

// 2x stress + verify + reinit

void launch_compute_stress_2x_verify_reinit_float(float *data, int *faulty,
                                                  size_t elements, int cycles,
                                                  uint32_t seed,
                                                  cudaStream_t stream);
void launch_compute_stress_2x_verify_reinit_double(double *data, int *faulty,
                                                   size_t elements, int cycles,
                                                   uint32_t seed,
                                                   cudaStream_t stream);
} // namespace gpuburn
