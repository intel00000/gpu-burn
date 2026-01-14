// gemm_kernels.h
#pragma once
#include <cstddef>
#include <cuda_runtime.h>

namespace gpuburn {

// Scale p[i] = p[i] * mul + ad

void launch_scale_float(float *p, size_t n, float mul, float add,
                        cudaStream_t stream);
void launch_scale_double(double *p, size_t n, double mul, double add,
                         cudaStream_t stream);

// Validate results by checking consistency across iterations

void launch_compare_float(const float *C, int *faulty, size_t iters,
                          size_t elems, cudaStream_t stream);
void launch_compare_double(const double *C, int *faulty, size_t iters,
                           size_t elems, cudaStream_t stream);

} // namespace gpuburn
