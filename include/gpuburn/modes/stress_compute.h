// stress_compute.h
#pragma once
#include "gpuburn/config.h"
#include "gpuburn/stress_mode.h"
#include <cuda_runtime.h>

template <typename T> class StressCompute final : public StressMode {
  public:
    StressCompute() = default;
    ~StressCompute() override { release(); }

    std::string_view name() const override { return "compute"; }
    void init(DeviceContext &dev, const StressConfig &cfg) override;
    StepResult step(DeviceContext &dev) override;
    PerfModel perf_model() const override;

  private:
    int device_id_ = -1;

    size_t elements_ = 0;     // elements per half
    size_t buffer_bytes_ = 0; // buffer size (2x elements)
    int iterations_ = 1000;   // iters to apply per step
    uint32_t seed_ = 42;      // increments each step

    T *dData_ = nullptr;
    int *dFaulty_ = nullptr;
    int *hFaultyPinned_ = nullptr;

    cudaEvent_t done_evt_ = nullptr;

    void release() noexcept;
};
