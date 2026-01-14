// stress_gemm.h
#pragma once
#include "gpuburn/config.h"
#include "gpuburn/stress_mode.h"
#include <cuda_runtime.h>

template <typename T> class StressGemm final : public StressMode {
  public:
    StressGemm() = default;
    ~StressGemm() override { release(); }

    std::string_view name() const override { return "gemm"; }
    void init(DeviceContext &dev, const StressConfig &cfg) override;
    StepResult step(DeviceContext &dev) override;
    PerfModel perf_model() const override;

  private:
    int device_id_ = -1;
    bool use_tensor_ = false;

    size_t size_ = 0;
    size_t elems_ = 0;
    size_t mat_bytes_ = 0;
    size_t iters_ = 0;

    T *dA_ = nullptr;
    T *dB_ = nullptr;
    T *dC_ = nullptr;
    int *dFaulty_ = nullptr;
    int *hFaultyPinned_ = nullptr;

    cudaEvent_t done_evt_ = nullptr;

    void release() noexcept;
};
