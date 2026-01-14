// platform.h
// Platform-specific utilities
#pragma once
#include <atomic>
#include <string>

namespace gpuburn {

bool enable_ansi();

std::string sformat(const char *fmt, ...);

void install_signal_handlers(std::atomic<bool> &running);

} // namespace gpuburn
