// stress_mode.h
// Stress testing mode interface
#pragma once
#include <cstdint>
#include <string_view>

struct DeviceContext;

struct StepResult {
    uint64_t units = 0;
    uint64_t errors = 0; // errors detected in this step
};

enum class MetricKind { Gflops, Gbytes };

struct PerfModel {
    MetricKind kind = MetricKind::Gflops;
    double per_unit = 0.0; // FLOPs per unit, or bytes per unit
};

struct StressConfig {
    bool use_doubles = false;
    bool use_tensor = false;
    int64_t use_bytes = 0; // 0 use default frac, < 0 mean %, > 0 mean MB
    int run_seconds = 10;
    size_t matrix_size = 8192; // Default matrix size (NxN)
};

class StressMode {
  public:
    virtual ~StressMode() = default;
    virtual std::string_view name() const = 0;
    // initialize with device context and config
    virtual void init(DeviceContext &dev, const StressConfig &cfg) = 0;
    // perform one step of work
    virtual StepResult step(DeviceContext &dev) = 0;
    // get performance model for reporting purposes
    virtual PerfModel perf_model() const = 0;

  protected:
    // Helper to resolve memory usage from config
    // 0 = use default fraction, < 0 = percentage, > 0 = MB
    size_t resolve_use_bytes(DeviceContext &dev, int64_t use_bytes_arg);
};
