// device_context.cpp
#include "gpuburn/device_context.h"
#include "gpuburn/errors.h"

void DeviceContext::init(int dev_id, bool use_tensor, bool use_doubles) {
    device_id = dev_id;
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice");
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");

    cublas_check(cublasCreate(&cublas), "cublasCreate");
    cublas_check(cublasSetStream(cublas, stream), "cublasSetStream");

    if (use_tensor && !use_doubles) {
        cublas_check(cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH),
                     "cublasSetMathMode(TF32)");
    }

    curand_check(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_DEFAULT),
                 "curandCreateGenerator");
    curand_check(curandSetPseudoRandomGeneratorSeed(rng, 42ULL),
                 "curandSetSeed");
    curand_check(curandSetStream(rng, stream), "curandSetStream");
}

void DeviceContext::shutdown() {
    if (cublas) {
        cublasDestroy(cublas);
        cublas = nullptr;
    }
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
    if (rng) {
        curandDestroyGenerator(rng);
        rng = nullptr;
    }
}

void DeviceContext::mem_info(size_t &free_bytes, size_t &total_bytes) const {
    cuda_check(cudaSetDevice(device_id), "cudaSetDevice(mem_info)");
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo");
}
