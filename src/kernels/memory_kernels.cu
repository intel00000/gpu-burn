// memory_kernels.cu
// Memory stress kernels - Sequential, random, and thrashing
#include "gpuburn/config.h"
#include "gpuburn/kernels/memory_kernels.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <type_traits>

namespace cg = cooperative_groups;

namespace gpuburn {

template <typename T> struct MemoryTraits;
template <> struct MemoryTraits<float> {
    using Scalar = float;
    using Vec4 = float4;
    static constexpr float eps = kEpsilonF;
    static __device__ float abs(float x) { return fabsf(x); }
    static __device__ bool is_bad(float x) { return isnan(x) || isinf(x); }
};
template <> struct MemoryTraits<double> {
    using Scalar = double;
    using Vec4 = double4;
    static constexpr double eps = kEpsilonD;
    static __device__ double abs(double x) { return fabs(x); }
    static __device__ bool is_bad(double x) { return isnan(x) || isinf(x); }
};

// Sequential memory access
template <typename T>
__global__ void memorySequential(T *__restrict__ data, size_t elements, int cycles) {
    const size_t idx =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    using Tr = MemoryTraits<T>;
    using V4 = typename Tr::Vec4;
    const size_t vec_elems = elements / 4;
    const size_t tail_idx = vec_elems * 4 + idx;
    for (int cycle = 0; cycle < cycles; cycle++) {
        // Vectorized access
        if (idx < vec_elems) {
            V4 *data4 = reinterpret_cast<V4 *>(data);
            V4 v = data4[idx];
            v.x = v.x * (T)1.5 + (T)0.5;
            v.y = v.y * (T)1.5 + (T)0.5;
            v.z = v.z * (T)1.5 + (T)0.5;
            v.w = v.w * (T)1.5 + (T)0.5;
            data4[idx] = v;
        }
        // Tail elements
        if (tail_idx < elements) {
            T v = data[tail_idx];
            data[tail_idx] = v * (T)1.5 + (T)0.5;
        }
    }
}

// Stride access pattern - read-only cache thrashing
template <typename T>
__global__ void memoryStride(const T *__restrict__ data, size_t elements,
                             size_t stride, int cycles) {
    const size_t idx =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= elements)
        return;
    T acc = (T)0;
    size_t pos = idx;
    for (int cycle = 0; cycle < cycles; cycle++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (pos < elements) {
                acc += data[pos];
                pos = (pos + stride) % elements;
            }
        }
    }
}

// Random access - init cuRAND states
__global__ void initRandStates(curandState *states, size_t n_states,
                               unsigned int seed) {
    int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid < n_states)
        curand_init(seed, tid, 0, &states[tid]);
}
// Random access - read-only with multiple cycles
template <typename T>
__global__ void memoryRandom(T *__restrict__ data,
                             curandState *__restrict__ states, size_t elems, int cycles) {
    const int tid = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int stride = (int)(gridDim.x * blockDim.x);
    curandState localState = states[tid];
    T total_acc = (T)0;

    for (int cycle = 0; cycle < cycles; cycle++) {
        for (size_t i = (size_t)tid; i < elems; i += (size_t)stride) {
            T acc = (T)0;
#pragma unroll
            for (int j = 0; j < 16; j++) {
                unsigned int r = curand(&localState);
                size_t rand_idx = (size_t)r % elems;
                acc += data[rand_idx];
            }
            total_acc += acc;
        }
    }
    states[tid] = localState;
}

// Validation kernels - compare two buffers
template <typename T>
__global__ void compareMemory(const T *__restrict__ ref,
                              const T *__restrict__ test,
                              int *__restrict__ faulty, size_t elems) {
    using Tr = MemoryTraits<T>;
    using V4 = typename Tr::Vec4;
    const size_t idx =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;

    int local_bad = 0;
    // 4 elements at a time
    const size_t vec_elems = elems / 4;
    if (idx < vec_elems) {
        V4 ref4 = reinterpret_cast<const V4 *>(ref)[idx];
        V4 test4 = reinterpret_cast<const V4 *>(test)[idx];
        if (Tr::abs(ref4.x - test4.x) > Tr::eps || Tr::is_bad(ref4.x) ||
            Tr::is_bad(test4.x))
            local_bad++;
        if (Tr::abs(ref4.y - test4.y) > Tr::eps || Tr::is_bad(ref4.y) ||
            Tr::is_bad(test4.y))
            local_bad++;
        if (Tr::abs(ref4.z - test4.z) > Tr::eps || Tr::is_bad(ref4.z) ||
            Tr::is_bad(test4.z))
            local_bad++;
        if (Tr::abs(ref4.w - test4.w) > Tr::eps || Tr::is_bad(ref4.w) ||
            Tr::is_bad(test4.w))
            local_bad++;
    }
    // Tail: remaining elements
    const size_t tail_idx = vec_elems * 4 + idx;
    if (tail_idx < elems) {
        T ref_val = ref[tail_idx];
        T test_val = test[tail_idx];
        if (Tr::abs(ref_val - test_val) > Tr::eps || Tr::is_bad(ref_val) ||
            Tr::is_bad(test_val))
            local_bad++;
    }
    // Error reduction and update using cooperative groups
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    int warp_total = cg::reduce(tile, local_bad, cg::plus<int>());
    if (tile.thread_rank() == 0 && warp_total > 0)
        atomicAdd(faulty, warp_total);
}

// Launch wrappers

void launch_memory_sequential_float(float *data, size_t elems, int cycles,
                                    cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)((elems / 4 + threads - 1) / threads);
    if (blocks == 0)
        blocks = 1;
    memorySequential<float><<<blocks, threads, 0, stream>>>(data, elems, cycles);
}
void launch_memory_sequential_double(double *data, size_t elems, int cycles,
                                     cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)((elems / 4 + threads - 1) / threads);
    if (blocks == 0)
        blocks = 1;
    memorySequential<double><<<blocks, threads, 0, stream>>>(data, elems, cycles);
}

void launch_memory_stride_float(float *data, size_t elements, size_t stride,
                                int cycles, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (int)((elements + threads - 1) / threads);
    memoryStride<float><<<blocks, threads, 0, stream>>>(data, elements, stride, cycles);
}
void launch_memory_stride_double(double *data, size_t elements, size_t stride,
                                 int cycles, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (int)((elements + threads - 1) / threads);
    memoryStride<double><<<blocks, threads, 0, stream>>>(data, elements, stride, cycles);
}

void launch_init_rand_states(void *states, size_t n_states, unsigned int seed,
                             cudaStream_t stream) {
    const int threads = 256;
    const int blocks = (int)((n_states + threads - 1) / threads);
    initRandStates<<<blocks, threads, 0, stream>>>((curandState *)states,
                                                   n_states, seed);
}
void launch_memory_random_float(float *data, void *states, size_t elems,
                                int cycles, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = 256;
    memoryRandom<float>
        <<<blocks, threads, 0, stream>>>(data, (curandState *)states, elems, cycles);
}
void launch_memory_random_double(double *data, void *states, size_t elems,
                                 int cycles, cudaStream_t stream) {
    const int threads = 256;
    const int blocks = 256;
    memoryRandom<double>
        <<<blocks, threads, 0, stream>>>(data, (curandState *)states, elems, cycles);
}

void launch_compare_memory_float(const float *ref, const float *test,
                                 int *faulty, size_t elems,
                                 cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)((elems / 4 + threads - 1) / threads);
    if (blocks == 0)
        blocks = 1;
    compareMemory<float>
        <<<blocks, threads, 0, stream>>>(ref, test, faulty, elems);
}
void launch_compare_memory_double(const double *ref, const double *test,
                                  int *faulty, size_t elems,
                                  cudaStream_t stream) {
    const int threads = 256;
    int blocks = (int)((elems / 4 + threads - 1) / threads);
    if (blocks == 0)
        blocks = 1;
    compareMemory<double>
        <<<blocks, threads, 0, stream>>>(ref, test, faulty, elems);
}
} // namespace gpuburn
