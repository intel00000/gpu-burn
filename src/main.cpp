#include "gpuburn/config.h"
#include "gpuburn/console.h"
#include "gpuburn/device_context.h"
#include "gpuburn/errors.h"
#include "gpuburn/modes/stress_compute.h"
#include "gpuburn/modes/stress_gemm.h"
#include "gpuburn/modes/stress_memory.h"
#include "gpuburn/nvml_monitor.h"
#include "gpuburn/platform.h"

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static void show_help() {
    std::printf(
        "GPU Burn - Multi-GPU CUDA stress test\n"
        "Usage: gpu-burn [OPTIONS] [TIME]\n\n"
        "Options:\n"
        "  -m X       Use X MB of memory\n"
        "  -m N%%      Use N%% of available GPU memory (default: %.0f%%)\n"
        "  -s N       Matrix size NxN (default: 8192, must be multiple of "
        "256)\n"
        "  -mode M    Stress mode: gemm, compute, memory (default: gemm)\n"
        "  -d         Use double precision (FP64)\n"
        "  -tc        Use tensor cores (TF32 for FP32, requires Ampere+)\n"
        "  -l         List GPUs and exit\n"
        "  -i N       Test only GPU N\n"
        "  -v         Verbose monitoring output\n"
        "  -p         Profile mode: run one iteration and exit\n"
        "  --no-nvml  Disable NVML GPU monitoring\n"
        "  -h         Show this help\n\n"
        "Stress Modes:\n"
        "  gemm:    Matrix multiply stress (default, -s sets matrix size)\n"
        "  compute: ALU+SFU compute stress with minimal memory bandwidth\n"
        "  memory:  Memory bandwidth + random access + PCIe transfers\n\n"
        "Monitoring:\n"
        "  Normal:  Shows temperature, power, throttling indicator (!)\n"
        "  Verbose: Shows temp, power, SM clock, mem clock, throttle "
        "reasons\n\n"
        "Examples:\n"
        "  gpu-burn 60                    # Test all GPUs for 60 seconds\n"
        "  gpu-burn -s 16384 300          # Larger matrices for max stress\n"
        "  gpu-burn -mode compute -v 60   # Compute stress with verbose "
        "output\n"
        "  gpu-burn -mode memory 3600     # Memory stress test for 1 hour\n",
        kDefaultUseMemFrac * 100.0);
}

// NNN MB, or NN% => returns negative percent, or 0 on error
static int64_t decode_usemem(const char *s) {
    char *end = nullptr;
    long long r = std::strtoll(s, &end, 10);
    if (end == s)
        return 0;
    if (*end == '%')
        return (end[1] == 0) ? -(int64_t)r : 0;
    return (*end == 0) ? (int64_t)r : 0;
}

static void list_gpus() {
    int count = 0;
    cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cuda_check(cudaGetDeviceProperties(&prop, i),
                   "cudaGetDeviceProperties");
        size_t free_b = 0, total_b = 0;
        cuda_check(cudaSetDevice(i), "cudaSetDevice");
        cuda_check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        std::printf("ID %d: %s, cc %d.%d, VRAM %zu MB (free %zu MB)\n", i,
                    prop.name, prop.major, prop.minor,
                    total_b / 1024ull / 1024ull, free_b / 1024ull / 1024ull);
    }
}

struct WorkerShared {
    std::atomic<uint64_t> total_units{0};
    std::atomic<uint64_t> total_errors{0};
    std::atomic<double> last_rate_gflops{0.0};
    std::atomic<int> alive{1};
    gpuburn::GPUMetrics metrics; // GPU monitoring metrics
    std::mutex metrics_mutex;    // Protect metrics updates
};

template <typename T>
static void run_workers(const std::vector<int> &devices,
                        const StressConfig &cfg, const std::string &mode_name,
                        std::atomic<bool> &running,
                        const std::chrono::steady_clock::time_point &deadline,
                        std::vector<WorkerShared> &shared,
                        std::vector<std::thread> &workers) {
    workers.reserve(devices.size());

    for (size_t idx = 0; idx < devices.size(); ++idx) {
        const int dev_id = devices[idx];

        workers.emplace_back([&, idx, dev_id, mode_name]() {
            try {
                DeviceContext dev;
                dev.init(dev_id, cfg.use_tensor, cfg.use_doubles);

                // Create the appropriate stress mode
                std::unique_ptr<StressMode> mode;
                if (mode_name == "gemm") {
                    mode = std::make_unique<StressGemm<T>>();
                } else if (mode_name == "compute") {
                    mode = std::make_unique<StressCompute<T>>();
                } else if (mode_name == "memory") {
                    mode = std::make_unique<StressMemory<T>>();
                }

                mode->init(dev, cfg);
                const PerfModel pm = mode->perf_model();

                auto last = std::chrono::steady_clock::now();
                while (running.load(std::memory_order_relaxed) &&
                       std::chrono::steady_clock::now() < deadline) {
                    StepResult r = mode->step(dev);

                    auto now = std::chrono::steady_clock::now();
                    const double dt =
                        std::chrono::duration<double>(now - last).count();
                    last = now;

                    double rate = 0.0;
                    if (dt > 0.0) {
                        rate =
                            (double)r.units * pm.per_unit / dt / 1e9; // GFLOP/s
                    }

                    shared[idx].total_units.fetch_add(
                        r.units, std::memory_order_relaxed);
                    shared[idx].total_errors.fetch_add(
                        r.errors, std::memory_order_relaxed);
                    shared[idx].last_rate_gflops.store(
                        rate, std::memory_order_relaxed);
                }

                dev.shutdown();
                shared[idx].alive.store(0, std::memory_order_relaxed);
            } catch (const std::exception &e) {
                std::fprintf(stderr, "GPU %d worker died: %s\n", dev_id,
                             e.what());
                shared[idx].alive.store(0, std::memory_order_relaxed);
            }
        });
    }
}

int main(int argc, char **argv) {
    StressConfig cfg;
    cfg.run_seconds = kDefaultRunSeconds;
    int device_id = -1;
    bool verbose = false;
    bool enable_nvml = true;
    std::string mode_name = "gemm";

    // Simple parsing:
    // - flags consume themselves (+ optional next token)
    // - any non-flag token is treated as TIME (last one wins)
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];

        if (std::strcmp(a, "-h") == 0) {
            show_help();
            return 0;
        }
        if (std::strcmp(a, "-l") == 0) {
            list_gpus();
            return 0;
        }
        if (std::strcmp(a, "-d") == 0) {
            cfg.use_doubles = true;
            continue;
        }
        if (std::strcmp(a, "-tc") == 0) {
            cfg.use_tensor = true;
            continue;
        }
        if (std::strcmp(a, "-v") == 0) {
            verbose = true;
            continue;
        }
        if (std::strcmp(a, "--no-nvml") == 0) {
            enable_nvml = false;
            continue;
        }
        if (std::strcmp(a, "-mode") == 0) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "Syntax error near -mode\n");
                return 2;
            }
            const char *val = argv[++i];
            if (std::strcmp(val, "gemm") != 0 &&
                std::strcmp(val, "compute") != 0 &&
                std::strcmp(val, "memory") != 0) {
                std::fprintf(
                    stderr,
                    "Invalid mode '%s'. Must be: gemm, compute, or memory\n",
                    val);
                return 2;
            }
            mode_name = val;
            continue;
        }
        if (std::strncmp(a, "-m", 2) == 0) {
            const char *val =
                a[2] ? (a + 2) : ((i + 1 < argc) ? argv[++i] : nullptr);
            if (!val) {
                std::fprintf(stderr, "Syntax error near -m\n");
                return 2;
            }
            const int64_t r = decode_usemem(val);
            if (r == 0) {
                std::fprintf(stderr, "Syntax error near -m\n");
                return 2;
            }
            cfg.use_bytes = r;
            continue;
        }
        if (std::strncmp(a, "-i", 2) == 0) {
            const char *val =
                a[2] ? (a + 2) : ((i + 1 < argc) ? argv[++i] : nullptr);
            if (!val) {
                std::fprintf(stderr, "Syntax error near -i\n");
                return 2;
            }
            device_id = std::atoi(val);
            continue;
        }
        if (std::strncmp(a, "-s", 2) == 0) {
            const char *val =
                a[2] ? (a + 2) : ((i + 1 < argc) ? argv[++i] : nullptr);
            if (!val) {
                std::fprintf(stderr, "Syntax error near -s\n");
                return 2;
            }
            size_t size = (size_t)std::atoi(val);
            if (size < 1024 || size > 32768 || size % 256 != 0) {
                std::fprintf(
                    stderr,
                    "Matrix size must be 1024-32768 and multiple of 256\n");
                return 2;
            }
            cfg.matrix_size = size;
            continue;
        }

        if (a[0] == '-') {
            continue; // ignore unknown flag
        }
        cfg.run_seconds = std::atoi(a);
    }

    int dev_count = 0;
    cuda_check(cudaGetDeviceCount(&dev_count), "cudaGetDeviceCount");
    if (dev_count <= 0) {
        std::fprintf(stderr, "No CUDA devices\n");
        return 3;
    }

    std::vector<int> devices;
    if (device_id >= 0) {
        if (device_id >= dev_count) {
            std::fprintf(stderr, "Invalid device id %d (count=%d)\n", device_id,
                         dev_count);
            return 3;
        }
        devices.push_back(device_id);
    } else {
        devices.reserve((size_t)dev_count);
        for (int i = 0; i < dev_count; ++i)
            devices.push_back(i);
    }

    std::atomic<bool> running{true};
    gpuburn::install_signal_handlers(running);

    // Initialize NVML for GPU monitoring
    gpuburn::NVMLMonitor nvml;
    bool nvml_available = false;
    if (enable_nvml) {
        try {
            nvml.init();
            nvml_available = nvml.is_available();
            if (nvml_available && verbose) {
                std::printf("NVML initialized successfully\n");
            }
        } catch (const std::exception &e) {
            std::fprintf(stderr, "NVML init failed: %s\n", e.what());
        }
    }

    std::vector<WorkerShared> shared(devices.size());
    std::vector<std::thread> workers;

    const auto start_time = std::chrono::steady_clock::now();
    const auto deadline = start_time + std::chrono::seconds(cfg.run_seconds);

    // Start workers with selected mode
    if (cfg.use_doubles) {
        run_workers<double>(devices, cfg, mode_name, running, deadline, shared,
                            workers);
    } else {
        run_workers<float>(devices, cfg, mode_name, running, deadline, shared,
                           workers);
    }

    // Give workers time to initialize and print their messages
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::printf("\n");
    std::fflush(stdout);

    const int block_lines = verbose ? (1 + (int)devices.size()) : 1;
    gpuburn::ConsoleTUI tui(block_lines);

    // Reporter loop
    while (std::chrono::steady_clock::now() < deadline) {
        bool any_alive = false;
        for (auto &s : shared) {
            if (s.alive.load(std::memory_order_relaxed)) {
                any_alive = true;
                break;
            }
        }
        if (!any_alive)
            break;

        const auto now = std::chrono::steady_clock::now();
        const double elapsed =
            std::chrono::duration<double>(now - start_time).count();
        double pct = (cfg.run_seconds > 0)
                         ? (elapsed / (double)cfg.run_seconds * 100.0)
                         : 100.0;
        if (pct > 100.0)
            pct = 100.0;

        if (nvml_available) {
            for (size_t i = 0; i < devices.size(); ++i) {
                gpuburn::GPUMetrics m;
                nvml.query_metrics(devices[i], m);
                std::lock_guard<std::mutex> lock(shared[i].metrics_mutex);
                shared[i].metrics = m;
            }
        }

        // Build display lines
        std::vector<std::string> lines;
        lines.reserve((size_t)block_lines);

        // Summary/progress line
        std::string proc;
        proc.reserve(256);
        const char *unit_str = (mode_name == "memory") ? "GB/s" : "Gflop/s";
        for (size_t i = 0; i < shared.size(); ++i) {
            const auto tot =
                shared[i].total_units.load(std::memory_order_relaxed);
            const auto rate =
                shared[i].last_rate_gflops.load(std::memory_order_relaxed);

            proc += gpuburn::sformat("%llu (%.0f %s)", (unsigned long long)tot,
                                     rate, unit_str);
            if (i + 1 != shared.size())
                proc += " - ";
        }

        std::string errs;
        errs.reserve(64);
        for (size_t i = 0; i < shared.size(); ++i) {
            const auto err =
                shared[i].total_errors.load(std::memory_order_relaxed);
            errs += gpuburn::sformat("%llu", (unsigned long long)err);
            if (i + 1 != shared.size())
                errs += " - ";
        }

        if (!verbose) {
            std::string therm;
            therm.reserve(64);
            for (size_t i = 0; i < shared.size(); ++i) {
                if (nvml_available) {
                    gpuburn::GPUMetrics m;
                    {
                        std::lock_guard<std::mutex> lock(
                            shared[i].metrics_mutex);
                        m = shared[i].metrics;
                    }
                    therm += gpuburn::sformat("%uC %uW%s", m.temperature_c,
                                              m.power_mw / 1000,
                                              m.is_throttling ? "!" : "");
                } else
                    therm += "--";
                if (i + 1 != shared.size())
                    therm += " - ";
            }
            lines.push_back(
                gpuburn::sformat("%.1f%%  proc'd: %s  errors: %s  %s", pct,
                                 proc.c_str(), errs.c_str(), therm.c_str()));
        } else {
            lines.push_back(gpuburn::sformat("%.1f%%  proc'd: %s  errors: %s",
                                             pct, proc.c_str(), errs.c_str()));
        }

        // Verbose reporting
        if (verbose) {
            for (size_t i = 0; i < shared.size(); ++i) {
                if (nvml_available) {
                    gpuburn::GPUMetrics m;
                    {
                        std::lock_guard<std::mutex> lock(
                            shared[i].metrics_mutex);
                        m = shared[i].metrics;
                    }
                    std::string line = gpuburn::sformat(
                        "  GPU %d: %uC, %uW, SM:%uMHz, Mem:%uMHz", devices[i],
                        m.temperature_c, m.power_mw / 1000, m.sm_clock_mhz,
                        m.mem_clock_mhz);
                    if (m.is_throttling) {
                        line += gpuburn::sformat(" [THROTTLE: %s]",
                                                 m.throttle_reasons.c_str());
                    }
                    lines.push_back(std::move(line));
                } else {
                    lines.push_back(gpuburn::sformat(
                        "  GPU %d: monitoring unavailable", devices[i]));
                }
            }
        }
        tui.draw(lines);
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kDefaultReportPeriodMs));
    }

    running.store(false, std::memory_order_relaxed);
    for (auto &t : workers)
        t.join();

    std::printf("\nDone.\n");
    return 0;
}
