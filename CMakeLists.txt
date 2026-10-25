cmake_minimum_required(VERSION 3.18)
project(cuda_gemm_optimization LANGUAGES CXX CUDA)

# Configurar padrão C++ e CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Habilitar otimizações do compilador (-O3)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Diretórios de cabeçalho (includes)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Localizar o pacote cuBLAS
find_package(CUDAToolkit REQUIRED)

# Definir o executável do projeto
add_executable(gemm_bench
    src/main.cu
    include/kernels/00_naive.cuh
    include/kernels/01_shared_mem.cuh
    include/kernels/02_float4_padded.cuh
    include/kernels/03_double_buffering.cuh
)

# Vincular as bibliotecas do CUDA (cuBLAS)
target_link_libraries(gemm_bench PRIVATE CUDA::cublas CUDA::cudart)

# Propriedades de arquitetura CUDA (T4 = sm_75, L4 = sm_89, A100 = sm_80)
set_property(TARGET gemm_bench PROPERTY CUDA_ARCHITECTURES 75 80 89)

message(STATUS "Configuração do CMake concluída para CUDA GEMM Optimization.")
