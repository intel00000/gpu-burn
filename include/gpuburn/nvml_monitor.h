// nvml_monitor.h
// GPU monitoring interface
#pragma once

#include <nvml.h>
#include <string>
#include <vector>

namespace gpuburn {

// Structure to hold GPU metrics
struct GPUMetrics {
    unsigned int temperature_c = 0;
    unsigned int power_mw = 0;
    unsigned int sm_clock_mhz = 0;
    unsigned int mem_clock_mhz = 0;
    bool is_throttling = false;
    std::string throttle_reasons;
};

class NVMLMonitor {
  public:
    NVMLMonitor() = default;
    ~NVMLMonitor();

    // Initialize and enumerate devices
    void init();

    // Shutdown
    void shutdown();

    // Query metrics using a device index
    void query_metrics(int device_id, GPUMetrics &out);

    // Check if the interface is available and initialized
    bool is_available() const { return initialized_; }

  private:
    bool initialized_ = false;
    std::vector<nvmlDevice_t> devices_;

    // Helper to convert NVML throttle reasons
    std::string get_throttle_reasons(nvmlDevice_t device);
};

} // namespace gpuburn
