// compute_kernels.cu
#include "gpuburn/config.h"
#include "gpuburn/kernels/compute_kernels.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>
#include <type_traits>

namespace cg = cooperative_groups;

namespace gpuburn {

#ifndef STRESS_LANES
#define STRESS_LANES 4
#endif

template <typename T> struct ComputeTraits;
template <> struct ComputeTraits<float> {
    using Scalar = float;
    using Vec4 = float4;
    using HashT = uint32_t;

    static constexpr float eps = kEpsilonF;

    static __device__ float abs(float x) { return fabsf(x); }
    static __device__ bool is_bad(float x) { return isnan(x) || isinf(x); }

    static __device__ float s(float x) { return sinf(x); }
    static __device__ float c(float x) { return cosf(x); }
    static __device__ float e(float x) { return expf(x); }
    static __device__ float l(float x) { return logf(x); }
    static __device__ float q(float x) { return sqrtf(x); }
    static __device__ float t(float x) { return tanhf(x); }
    // fnv-1a hash functions for scattering
    static __device__ uint32_t fnv1a(uint32_t x) {
        uint32_t h = 2166136261u;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            uint32_t b = (x >> (i * 8)) & 0xFFu;
            h ^= b;
            h *= 16777619u;
        }
        return h;
    }
    // Convert hash to [0,1) float
    static __device__ float to_unit(uint32_t h) {
        uint32_t bits = (h & 0x007FFFFFu) | 0x3F800000u;
        return __int_as_float((int)bits) - 1.0f;
    }
};
template <> struct ComputeTraits<double> {
    using Scalar = double;
    using Vec4 = double4;
    using HashT = uint64_t;
    static constexpr double eps = kEpsilonD;
    static __device__ double abs(double x) { return fabs(x); }
    static __device__ bool is_bad(double x) { return isnan(x) || isinf(x); }
    static __device__ double s(double x) { return sin(x); }
    static __device__ double c(double x) { return cos(x); }
    static __device__ double e(double x) { return exp(x); }
    static __device__ double l(double x) { return log(x); }
    static __device__ double q(double x) { return sqrt(x); }
    static __device__ double t(double x) { return tanh(x); }
    static __device__ uint64_t fnv1a(uint64_t x) {
        uint64_t h = 14695981039346656037ull;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            uint64_t b = (x >> (i * 8)) & 0xFFull;
            h ^= b;
            h *= 1099511628211ull;
        }
        return h;
    }
    static __device__ double to_unit(uint64_t h) {
        uint64_t bits = (h & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
        return __longlong_as_double((long long)bits) - 1.0;
    }
};

// Stress step
template <typename T>
__device__ __forceinline__ T stress_step(T v, T phase, T offset) {
    using Tr = ComputeTraits<T>;
    T a = v + phase + offset;
    T s = Tr::s(a);
    T c = Tr::c(a);
    T sc = s * c;
    T e = Tr::e(sc * (T)0.001);
    T l = Tr::l(Tr::abs(sc) + (T)1.0);
    v = e - l;
    v = Tr::q(Tr::abs(v) + (T)0.001);
    v = Tr::t(v);
    return v;
}

// computeVerifyAndReinit: check data[0:elements) and data[elements:2*elements)
template <typename T>
__global__ void computeVerifyAndReinit(T *__restrict__ data,
                                       int *__restrict__ faultyElems,
                                       size_t elements, uint32_t seed) {
    const size_t idx =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    using Tr = ComputeTraits<T>;
    using V4 = typename Tr::Vec4;
    using H = typename Tr::HashT;

    int local_bad = 0;
    // 4 elements at a time
    const size_t vec_elems = elements / 4;
    if (idx < vec_elems) {
        V4 *a4 = reinterpret_cast<V4 *>(data);
        V4 *b4 = reinterpret_cast<V4 *>(data + elements); // second copy
        V4 r1 = a4[idx];
        V4 r2 = b4[idx];
        if (Tr::abs(r1.x - r2.x) > Tr::eps || Tr::is_bad(r1.x) ||
            Tr::is_bad(r2.x))
            local_bad++;
        if (Tr::abs(r1.y - r2.y) > Tr::eps || Tr::is_bad(r1.y) ||
            Tr::is_bad(r2.y))
            local_bad++;
        if (Tr::abs(r1.z - r2.z) > Tr::eps || Tr::is_bad(r1.z) ||
            Tr::is_bad(r2.z))
            local_bad++;
        if (Tr::abs(r1.w - r2.w) > Tr::eps || Tr::is_bad(r1.w) ||
            Tr::is_bad(r2.w))
            local_bad++;
        // Reinit all 4 lanes
        const H s = (H)seed;
        const H base = (H)(idx * 4);
        H h0 = Tr::fnv1a((base + (H)0) ^ s);
        H h1 = Tr::fnv1a((base + (H)1) ^ s);
        H h2 = Tr::fnv1a((base + (H)2) ^ s);
        H h3 = Tr::fnv1a((base + (H)3) ^ s);
        V4 v;
        v.x = (Tr::to_unit(h0) * (T)2.0 - (T)1.0) * (T)10.0;
        v.y = (Tr::to_unit(h1) * (T)2.0 - (T)1.0) * (T)10.0;
        v.z = (Tr::to_unit(h2) * (T)2.0 - (T)1.0) * (T)10.0;
        v.w = (Tr::to_unit(h3) * (T)2.0 - (T)1.0) * (T)10.0;
        a4[idx] = v;
        b4[idx] = v;
    }
    // Tail: remaining elements
    const size_t tail_idx = vec_elems * 4 + idx;
    if (tail_idx < elements) {
        T r1 = data[tail_idx];
        T r2 = data[tail_idx + elements];
        if (Tr::abs(r1 - r2) > Tr::eps || Tr::is_bad(r1) || Tr::is_bad(r2))
            local_bad++;
        H h = Tr::fnv1a(((H)tail_idx) ^ (H)seed);
        T v = (Tr::to_unit(h) * (T)2.0 - (T)1.0) * (T)10.0;
        data[tail_idx] = v;
        data[tail_idx + elements] = v;
    }
    // Error reduction and update using cooperative groups
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    int warp_total = cg::reduce(tile, local_bad, cg::plus<int>());
    if (tile.thread_rank() == 0 && warp_total > 0)
        atomicAdd(faultyElems, warp_total);
}

// computeStress2xVerifyReinit: stress twice, verify, reinit
template <typename T>
__global__ void
computeStress2xVerifyReinit(T *__restrict__ data, int *__restrict__ faultyElems,
                            size_t elements, int cycles, uint32_t seed) {
    using Tr = ComputeTraits<T>;
    using H = typename Tr::HashT;
    const size_t idx =
        (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (idx >= elements)
        return;

    // Starting value
    T x = data[idx], x1 = x, x2 = x;
    H hash = Tr::fnv1a((H)idx);
    T bp = (T)(hash & (H)0xFFFF);

    // lane params
#if defined(STRESS_LANES) && STRESS_LANES == 4
    T p0 = bp + (T)0.0, p1 = bp + (T)8192.0, p2 = bp + (T)16384.0,
      p3 = bp + (T)24576.0;
    T a0 = x1 + (T)0.01, a1 = x1 + (T)0.11, a2 = x1 + (T)0.21,
      a3 = x1 + (T)0.31;
    T b0 = x2 + (T)0.01, b1 = x2 + (T)0.11, b2 = x2 + (T)0.21,
      b3 = x2 + (T)0.31;
    for (int i = 0; i < cycles; i++) {
        T offset = (T)i * (T)0.001;
        a0 = stress_step<T>(a0, p0, offset);
        a1 = stress_step<T>(a1, p1, offset);
        a2 = stress_step<T>(a2, p2, offset);
        a3 = stress_step<T>(a3, p3, offset);
        b0 = stress_step<T>(b0, p0, offset);
        b1 = stress_step<T>(b1, p1, offset);
        b2 = stress_step<T>(b2, p2, offset);
        b3 = stress_step<T>(b3, p3, offset);
    }
    T r1 = (T)0.25 * (a0 + a1 + a2 + a3);
    T r2 = (T)0.25 * (b0 + b1 + b2 + b3);
#elif defined(STRESS_LANES) && STRESS_LANES == 2
    T p0 = bp + (T)0.0, p1 = bp + (T)16384.0;
    T a0 = x1 + (T)0.01, a1 = x1 + (T)0.21;
    T b0 = x2 + (T)0.01, b1 = x2 + (T)0.21;
    for (int i = 0; i < cycles; i++) {
        T offset = (T)i * (T)0.001;
        a0 = stress_step<T>(a0, p0, offset);
        a1 = stress_step<T>(a1, p1, offset);
        b0 = stress_step<T>(b0, p0, offset);
        b1 = stress_step<T>(b1, p1, offset);
    }
    T r1 = (T)0.5 * (a0 + a1), r2 = (T)0.5 * (b0 + b1);
#else
    T p0 = bp + (T)0.0;
    T a0 = x1 + (T)0.01;
    T b0 = x2 + (T)0.01;
    for (int i = 0; i < cycles; i++) {
        T offset = (T)i * (T)0.001;
        a0 = stress_step<T>(a0, p0, offset);
        b0 = stress_step<T>(b0, p0, offset);
    }
    T r1 = a0, r2 = b0;
#endif
    // Check results
    bool bad = (Tr::abs(r1 - r2) > Tr::eps) || Tr::is_bad(r1) || Tr::is_bad(r2);
    unsigned int mask = __ballot_sync(__activemask(), bad);
    if ((threadIdx.x & (warpSize - 1)) == 0 && mask != 0) {
        atomicAdd(faultyElems, __popc(mask));
    }
    // Reinit for next step
    H hh = Tr::fnv1a(((H)idx) ^ (H)seed);
    T v = (Tr::to_unit(hh) * (T)2.0 - (T)1.0) * (T)10.0;
    data[idx] = v;
    data[idx + elements] = v;
}

// ----------------------------------------------------------------------------
// Launchers templates
// ----------------------------------------------------------------------------

template <typename T>
static inline void launch_compute_verify_reinit(T *data, int *faulty,
                                                size_t elements, uint32_t seed,
                                                cudaStream_t stream) {
    constexpr int threads = 512;
    int blocks = (int)((elements / 4 + threads - 1) / threads);
    if (blocks == 0)
        blocks = 1;
    computeVerifyAndReinit<T>
        <<<blocks, threads, 0, stream>>>(data, faulty, elements, seed);
}
template <typename T>
static inline void
launch_compute_stress_2x_verify_reinit(T *data, int *faulty, size_t elements,
                                       int cycles, uint32_t seed,
                                       cudaStream_t stream) {
    constexpr int threads = 512;
    const int blocks = (int)((elements + threads - 1) / threads);
    computeStress2xVerifyReinit<T>
        <<<blocks, threads, 0, stream>>>(data, faulty, elements, cycles, seed);
}

// ----------------------------------------------------------------------------
// Public wrappers
// ----------------------------------------------------------------------------

void launch_compute_verify_reinit_float(float *data, int *faulty,
                                        size_t elements, uint32_t seed,
                                        cudaStream_t stream) {
    launch_compute_verify_reinit<float>(data, faulty, elements, seed, stream);
}
void launch_compute_verify_reinit_double(double *data, int *faulty,
                                         size_t elements, uint32_t seed,
                                         cudaStream_t stream) {
    launch_compute_verify_reinit<double>(data, faulty, elements, seed, stream);
}

void launch_compute_stress_2x_verify_reinit_float(float *data, int *faulty,
                                                  size_t elements, int cycles,
                                                  uint32_t seed,
                                                  cudaStream_t stream) {
    launch_compute_stress_2x_verify_reinit<float>(data, faulty, elements,
                                                  cycles, seed, stream);
}
void launch_compute_stress_2x_verify_reinit_double(double *data, int *faulty,
                                                   size_t elements, int cycles,
                                                   uint32_t seed,
                                                   cudaStream_t stream) {
    launch_compute_stress_2x_verify_reinit<double>(data, faulty, elements,
                                                   cycles, seed, stream);
}

} // namespace gpuburn
