# GPU-Burn

A multi-GPU CUDA stress testing tool with monitoring and multiple modes.

## Features

- **Multiple Modes**: GEMM (matrix multiply), Compute (ALU+SFU), Memory (bandwidth + random access)
- **GPU Monitoring**: Temperature, power, clocks, and throttling detection via NVML
- **Multi-GPU Support**: Test all GPUs simultaneously with independent error tracking
- **Configurable Parameters**: Matrix sizes, memory usage, precision (FP32/FP64), tensor cores
- **Cross-Platform**: Windows and Linux support

## Building

### Prerequisites

- CMake 3.24+
- CUDA Toolkit 11.0+
- NVML library (included with NVIDIA drivers)

### Build Instructions

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

## Prebuilt Binaries

Download from [Releases](https://github.com/intel00000/gpu-burn/releases). Binaries are dynamically linked and require matching CUDA installed, check your version with `nvcc --version` or `nvidia-smi`.

| Binary | Requires |
| -------- | ---------- |
| `gpu-burn-linux-cuda13.1.0.tar.gz` | CUDA 13.1 |
| `gpu-burn-linux-cuda12.8.1.tar.gz` | CUDA 12.8 |
| `gpu-burn-linux-cuda11.8.0.tar.gz` | CUDA 11.8 |
| `gpu-burn-windows-cuda13.1.0.exe` | CUDA 13.1 |
| `gpu-burn-windows-cuda12.8.1.exe` | CUDA 12.8 |

## Usage

```bash
# Basic usage - test all GPUs for 60 seconds
gpu-burn 60

# Use 90% of GPU memory instead of default 80%
gpu-burn -m 90% 300

# Test specific GPU
gpu-burn -i 0 120

# Verbose monitoring with detailed metrics
gpu-burn -v 60

# Different stress modes
gpu-burn -mode compute 120     # ALU+SFU compute stress
gpu-burn -mode memory 120      # Memory bandwidth stress
gpu-burn -mode gemm 120        # Matrix multiply stress (default)

# Larger matrices for maximum stress
gpu-burn -s 16384 300

# Double precision (FP64)
gpu-burn -d 300

# Try tensor cores (if supported)
gpu-burn -tc 300
```

### Command-Line Options

| Option | Description |
| -------- | ------------- |
| `-m X` | Use X MB of GPU memory |
| `-m N%` | Use N% of available GPU memory (default: 80%) |
| `-s N` | Matrix size NxN for GEMM mode (default: 8192, must be multiple of 256) |
| `-mode M` | Stress mode: `gemm`, `compute`, `memory` (default: `gemm`) |
| `-d` | Use double precision (FP64) |
| `-tc` | Use tensor cores (only in GEMM mode; uses TF32 for FP32, requires Ampere+) |
| `-l` | List GPUs and exit |
| `-i N` | Test GPU index N only (0-based) |
| `-v` | Verbose output |
| `-h` | Show help |

## Stress Modes

### GEMM Mode (default)

Matrix multiplication stress using cuBLAS. `-s` option controls matrix size.

### Compute Mode

ALU and SFU (Special Function Unit) stress with minimal memory bandwidth requirements.

### Memory Mode

Memory stress testing with four patterns that cycle continuously (not well designed currently):

1. **Sequential Access** (with validation): Coalesced memory reads/writes
2. **Random Access** (read-only): Random memory access
3. **Stride Access** (read-only): Strided access to cause thrashing
4. **PCIe Transfer** (with validation): Host-Device transfers

## Monitoring Output

### Normal Mode

Use the same output format from the original gpu-burn:

```text
89.2%  proc'd: 1234 (7823.5 Gflop/s)  errors: 0  72C 245W
```

Shows: Progress %, iterations processed, performance, errors, temperature, power, throttling indicator (!)

### Verbose Mode

```text
89.2%  proc'd: 1234 (7823.5 Gflop/s)  errors: 0
GPU 0: 72C  245W  SM:1890MHz  Mem:5001MHz
```

Shows: Detailed per-GPU metrics including clock speeds and throttling reasons

## Error Detection

All modes include built-in validation (not exhaustive):

- **GEMM mode**: Compares matrix multiplication results against reference
- **Compute mode**: Validates ALU and SFU computation consistency
- **Memory mode**: Validates sequential and PCIe patterns every 4th iteration

## Credits

Based on the original [gpu-burn](https://github.com/wilicc/gpu-burn), rewritten with:

- NVML integration for real-time monitoring
- Multiple stress modes (GEMM, Compute, Memory)
- Cross-platform support (Windows/Linux)
