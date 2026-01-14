#include "gpuburn/nvml_monitor.h"
#include <cstdio>
#include <stdexcept>

namespace gpuburn {

NVMLMonitor::~NVMLMonitor() { shutdown(); }

void NVMLMonitor::init() {
    if (initialized_)
        return;

    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::fprintf(stderr, "NVML init failed: %s\n", nvmlErrorString(result));
        return;
    }

    // Enumerate devices
    unsigned int device_count = 0;
    result = nvmlDeviceGetCount(&device_count);
    if (result != NVML_SUCCESS) {
        std::fprintf(stderr, "NVML device count failed: %s\n",
                     nvmlErrorString(result));
        nvmlShutdown();
        return;
    }

    devices_.resize(device_count);
    for (unsigned int i = 0; i < device_count; ++i) {
        result = nvmlDeviceGetHandleByIndex(i, &devices_[i]);
        if (result != NVML_SUCCESS) {
            std::fprintf(stderr, "NVML get device %u failed: %s\n", i,
                         nvmlErrorString(result));
            devices_[i] = nullptr;
        }
    }

    initialized_ = true;
}

void NVMLMonitor::shutdown() {
    if (initialized_) {
        nvmlShutdown();
        initialized_ = false;
        devices_.clear();
    }
}

void NVMLMonitor::query_metrics(int device_id, GPUMetrics &out) {
    if (!initialized_ || device_id < 0 || device_id >= (int)devices_.size()) {
        return;
    }

    nvmlDevice_t device = devices_[device_id];
    if (device == nullptr) {
        return;
    }

    unsigned int temp = 0;
    if (nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp) ==
        NVML_SUCCESS) {
        out.temperature_c = temp;
    }
    unsigned int power = 0;
    if (nvmlDeviceGetPowerUsage(device, &power) == NVML_SUCCESS) {
        out.power_mw = power;
    }
    unsigned int sm_clock = 0;
    if (nvmlDeviceGetClock(device, NVML_CLOCK_SM, NVML_CLOCK_ID_CURRENT,
                           &sm_clock) == NVML_SUCCESS) {
        out.sm_clock_mhz = sm_clock;
    }
    unsigned int mem_clock = 0;
    if (nvmlDeviceGetClock(device, NVML_CLOCK_MEM, NVML_CLOCK_ID_CURRENT,
                           &mem_clock) == NVML_SUCCESS) {
        out.mem_clock_mhz = mem_clock;
    }
    out.throttle_reasons = get_throttle_reasons(device);
    out.is_throttling = !out.throttle_reasons.empty();
}

std::string NVMLMonitor::get_throttle_reasons(nvmlDevice_t device) {
    unsigned long long reasons = 0;
    nvmlReturn_t result =
        nvmlDeviceGetCurrentClocksThrottleReasons(device, &reasons);
    if (result != NVML_SUCCESS) {
        return "";
    }

    if (reasons == nvmlClocksThrottleReasonNone) {
        return "";
    }

    std::string throttle_str;
    if (reasons & nvmlClocksThrottleReasonGpuIdle)
        throttle_str += "GPU Idle ";
    if (reasons & nvmlClocksThrottleReasonApplicationsClocksSetting)
        throttle_str += "Software Clocks Limit ";
    if (reasons & nvmlClocksThrottleReasonSwPowerCap)
        throttle_str += "Software Power Limit ";
    if (reasons & nvmlClocksThrottleReasonHwSlowdown)
        throttle_str += "Hardware Slowdown (e.g. high temp) ";
    if (reasons & nvmlClocksThrottleReasonSyncBoost)
        throttle_str += "Sync Boost ";
    if (reasons & nvmlClocksThrottleReasonSwThermalSlowdown)
        throttle_str += "Software Thermal ";
    if (reasons & nvmlClocksThrottleReasonHwThermalSlowdown)
        throttle_str += "Hardware Thermal ";
    if (reasons & nvmlClocksThrottleReasonHwPowerBrakeSlowdown)
        throttle_str += "Hardware PowerBrake ";
    if (reasons & nvmlClocksThrottleReasonDisplayClockSetting)
        throttle_str += "DisplayClock Limit ";
    if (!throttle_str.empty()) {
        throttle_str.pop_back();
    }

    return throttle_str;
}

} // namespace gpuburn
