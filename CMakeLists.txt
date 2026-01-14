cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")

project(gpu_burn LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(gpu-burn
  src/main.cpp
  src/console.cpp
  src/platform.cpp
  src/device_context.cpp
  src/nvml_monitor.cpp
  src/stress_mode.cpp
  src/modes/stress_gemm.cpp
  src/modes/stress_compute.cpp
  src/modes/stress_memory.cpp
  src/kernels/compute_kernels.cu
  src/kernels/gemm_kernels.cu
  src/kernels/memory_kernels.cu
)

get_target_property(_arch gpu-burn CUDA_ARCHITECTURES)
message(STATUS "gpu-burn CUDA_ARCHITECTURES=${_arch}")

target_include_directories(gpu-burn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(gpu-burn PRIVATE CUDA::cudart CUDA::cublas CUDA::curand CUDA::nvml)

if (WIN32)
  target_compile_definitions(gpu-burn PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()
