// GEMM kernels: scaling and validation
#include "gpuburn/config.h"
#include "gpuburn/kernels/gemm_kernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <type_traits>

namespace cg = cooperative_groups;

namespace gpuburn {

template <typename T> struct GemmTraits;
template <> struct GemmTraits<float> {
    using Vec4 = float4;
    static __device__ __forceinline__ float abs(float x) { return fabsf(x); }
    static constexpr float eps = kEpsilonF;
};
template <> struct GemmTraits<double> {
    using Vec4 = double4;
    static __device__ __forceinline__ double abs(double x) { return fabs(x); }
    static constexpr double eps = kEpsilonD;
};

// -----------------------------------------------------------------------------
// Scale kernel: p[i] = p[i] * mul + add
// Vectorized by 4 with a small tail handled by the first few threads.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void scale_kernel(T *__restrict__ p, size_t n, T mul, T add) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    using V4 = typename GemmTraits<T>::Vec4;
    const size_t vec_n = n / 4;
    // 4 floats at a time
    if (idx < vec_n) {
        V4 *p4 = reinterpret_cast<V4 *>(p);
        V4 v = p4[idx];
        v.x = v.x * mul + add;
        v.y = v.y * mul + add;
        v.z = v.z * mul + add;
        v.w = v.w * mul + add;
        p4[idx] = v;
    }
    // remaining tail
    const size_t tail_idx = vec_n * 4 + idx;
    if (tail_idx < n) {
        p[tail_idx] = p[tail_idx] * mul + add;
    }
}

// -----------------------------------------------------------------------------
// Compare kernel: validates that all GEMM iterations produce consistent results
// Assumes C is laid out as: [iter0 elems][iter1 elems]...[iter(iters-1) elems]
// Vectorized by 4 within each iteration.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void compare_kernel(const T *__restrict__ C,
                               int *__restrict__ faulty, size_t iters,
                               size_t elems) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    using V4 = typename GemmTraits<T>::Vec4;
    const size_t vec_elems = elems / 4;
    int local_bad = 0;
    // Vector path: 4 elements at a time
    if (idx < vec_elems) {
        const size_t base0 = idx * 4;
        V4 prev = *reinterpret_cast<const V4 *>(C + base0); // previous
        for (size_t i = 1; i < iters; ++i) { // compare subsequent iterations
            const size_t base = i * elems + base0;
            V4 cur = *reinterpret_cast<const V4 *>(C + base);
            if (GemmTraits<T>::abs(cur.x - prev.x) > GemmTraits<T>::eps)
                local_bad++;
            if (GemmTraits<T>::abs(cur.y - prev.y) > GemmTraits<T>::eps)
                local_bad++;
            if (GemmTraits<T>::abs(cur.z - prev.z) > GemmTraits<T>::eps)
                local_bad++;
            if (GemmTraits<T>::abs(cur.w - prev.w) > GemmTraits<T>::eps)
                local_bad++;
            prev = cur;
        }
    }
    // Scalar tail
    const size_t tail_idx = vec_elems * 4 + idx;
    if (tail_idx < elems) {
        T prev = C[tail_idx];
        for (size_t i = 1; i < iters; ++i) {
            T cur = C[tail_idx + i * elems];
            if (GemmTraits<T>::abs(cur - prev) > GemmTraits<T>::eps)
                local_bad++;
            prev = cur;
        }
    }
    // Warp reduction + one atomic per warp
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    int warp_total = cg::reduce(tile, local_bad, cg::plus<int>());
    if (tile.thread_rank() == 0 && warp_total > 0)
        atomicAdd(faulty, warp_total);
}

// -----------------------------------------------------------------------------
// Launch templates
// -----------------------------------------------------------------------------
template <typename T>
static inline void launch_scale(T *p, size_t n, T mul, T add,
                                cudaStream_t stream) {
    constexpr int threads = 512;
    // Blocks size for vectorized loads (each processes 4 elements + tail)
    int blocks = (int)((n / 4 + threads - 1) / threads);
    if (blocks == 0)
        blocks = 1;
    scale_kernel<T><<<blocks, threads, 0, stream>>>(p, n, mul, add);
}
template <typename T>
static inline void launch_compare(const T *C, int *faulty, size_t iters,
                                  size_t elems, cudaStream_t stream) {
    if (iters < 2)
        return;
    constexpr int threads = 512;
    int blocks = (int)(((elems / 4) + threads - 1) / threads);
    if (blocks == 0)
        blocks = 1;
    compare_kernel<T><<<blocks, threads, 0, stream>>>(C, faulty, iters, elems);
}

// -----------------------------------------------------------------------------
// Public wrappers
// -----------------------------------------------------------------------------
void launch_scale_float(float *p, size_t n, float mul, float add,
                        cudaStream_t stream) {
    launch_scale<float>(p, n, mul, add, stream);
}
void launch_scale_double(double *p, size_t n, double mul, double add,
                         cudaStream_t stream) {
    launch_scale<double>(p, n, mul, add, stream);
}
void launch_compare_float(const float *C, int *faulty, size_t iters,
                          size_t elems, cudaStream_t stream) {
    launch_compare<float>(C, faulty, iters, elems, stream);
}
void launch_compare_double(const double *C, int *faulty, size_t iters,
                           size_t elems, cudaStream_t stream) {
    launch_compare<double>(C, faulty, iters, elems, stream);
}
} // namespace gpuburn
