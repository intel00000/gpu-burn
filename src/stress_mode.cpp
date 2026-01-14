// stress_mode.cpp
#include "gpuburn/stress_mode.h"
#include "gpuburn/config.h"
#include "gpuburn/device_context.h"

size_t StressMode::resolve_use_bytes(DeviceContext &dev,
                                     int64_t use_bytes_arg) {
    size_t free_b = 0, total_b = 0;
    dev.mem_info(free_b, total_b);

    if (use_bytes_arg == 0)
        return (size_t)((double)free_b * kDefaultUseMemFrac);
    if (use_bytes_arg < 0)
        return (size_t)((double)free_b * ((double)(-use_bytes_arg) / 100.0));
    return (size_t)use_bytes_arg * 1024ull * 1024ull; // MB
}
