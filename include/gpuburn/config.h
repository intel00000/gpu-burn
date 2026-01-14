// gpuburn/config.h
// Configuration constants
#pragma once

// Default memory usage fraction
constexpr double kDefaultUseMemFrac = 0.90;

// Compute mode memory limits
constexpr size_t kComputeMinBytes = 32ull * 1024 * 1024;      // 32 MB
constexpr size_t kComputeMaxBytes = 1024ull * 1024 * 1024;    // 1024 MB
constexpr size_t kComputeDefaultBytes = 128ull * 1024 * 1024; // 128 MB

// Memory mode PCIe host buffer limit
constexpr size_t kMaxPCIeBufferBytes = 128ull * 1024 * 1024; // 128 MB

// Validation epsilons
constexpr float kEpsilonF = 0.001f;
constexpr double kEpsilonD = 0.0000001;

// Reporting
constexpr int kDefaultRunSeconds = 15;
constexpr int kDefaultReportPeriodMs = 200;
