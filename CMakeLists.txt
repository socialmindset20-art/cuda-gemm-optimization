cmake_minimum_required(VERSION 3.18)
project(cuda_gemm_optimization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

include(FetchContent)

FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG        v3.5.0
    GIT_SHALLOW    TRUE
)

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cutlass)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/include
)

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_bench
    src/main.cu
    include/kernels/00_naive.cuh
    include/kernels/01_shared_mem.cuh
    include/kernels/02_float4_padded.cuh
    include/kernels/03_double_buffering.cuh
    include/kernels/04_fp16_half2.cuh
    include/kernels/05_wmma_tensor_cores.cuh
    include/kernels/06_generic_gemm.cuh
    include/kernels/07_cutlass_basic.cuh
    include/kernels/08_cutlass_tensor_op.cuh
)

target_link_libraries(gemm_bench PRIVATE CUDA::cublas CUDA::cudart nvidia::cutlass::cutlass)
set_property(TARGET gemm_bench PROPERTY CUDA_ARCHITECTURES 75 80 89)
