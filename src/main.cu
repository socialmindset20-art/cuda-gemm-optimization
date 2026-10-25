#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "kernels/00_naive.cuh"
#include "kernels/01_shared_mem.cuh"
#include "kernels/02_float4_padded.cuh"
#include "kernels/03_double_buffering.cuh"
#include "kernels/04_fp16_half2.cuh"
#include "kernels/05_wmma_tensor_cores.cuh"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "Erro CUDA: " << cudaGetErrorString(err) << " na linha " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "Erro cuBLAS na linha " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    }

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc >= 4) {
        M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]);
    }

    std::cout << "=========================================================" << std::endl;
    std::cout << " CUDA GEMM Optimization Benchmark (FP32 vs FP16 vs WMMA)" << std::endl;
    std::cout << " Dimensões: M = " << M << ", N = " << N << ", K = " << K << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    size_t size_A_fp16 = M * K * sizeof(__half);
    size_t size_B_fp16 = K * N * sizeof(__half);
    size_t size_C_fp32 = M * N * sizeof(float);

    std::vector<__half> h_A_fp16(M * K);
    std::vector<__half> h_B_fp16(K * N);

    for (int i = 0; i < M * K; ++i) h_A_fp16[i] = __float2half(static_cast<float>(rand() % 10) / 10.0f);
    for (int i = 0; i < K * N; ++i) h_B_fp16[i] = __float2half(static_cast<float>(rand() % 10) / 10.0f);

    __half *d_A_fp16, *d_B_fp16;
    float *d_C_wmma, *d_C_cublas_fp16;

    CUDA_CHECK(cudaMalloc(&d_A_fp16, size_A_fp16));
    CUDA_CHECK(cudaMalloc(&d_B_fp16, size_B_fp16));
    CUDA_CHECK(cudaMalloc(&d_C_wmma, size_C_fp32));
    CUDA_CHECK(cudaMalloc(&d_C_cublas_fp16, size_C_fp32));

    CUDA_CHECK(cudaMemcpy(d_A_fp16, h_A_fp16.data(), size_A_fp16, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_fp16, h_B_fp16.data(), size_B_fp16, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_flops = 2.0 * static_cast<double>(M) * N * K;
    const int num_iterations = 5;
    float ms = 0.0f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    __half alpha_fp16 = __float2half(1.0f), beta_fp16 = __float2half(0.0f);

    // Reference: cuBLAS HGEMM
    CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                            &alpha_fp16, d_B_fp16, N, d_A_fp16, K, &beta_fp16, reinterpret_cast<__half*>(d_C_cublas_fp16), N));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        CUBLAS_CHECK(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                                &alpha_fp16, d_B_fp16, N, d_A_fp16, K, &beta_fp16, reinterpret_cast<__half*>(d_C_cublas_fp16), N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms_cublas = ms / num_iterations;
    double gflops_cublas = (total_flops * 1e-9) / (avg_ms_cublas / 1000.0f);

    std::cout << "[Reference] cuBLAS HGEMM (FP16): " << std::fixed << std::setprecision(2) 
              << avg_ms_cublas << " ms | " << gflops_cublas << " GFLOPS\n" << std::endl;

    // Kernel 05: WMMA Tensor Cores
    {
        dim3 blockDim(WMMA_BLOCK_DIM_X, WMMA_BLOCK_DIM_Y);
        dim3 gridDim((N + WMMA_BN - 1) / WMMA_BN, (M + WMMA_BM - 1) / WMMA_BM);

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iterations; ++i) {
            gemm_05_wmma_tensor_cores<<<gridDim, blockDim>>>(M, N, K, d_A_fp16, d_B_fp16, d_C_wmma);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iterations;
        double gflops = (total_flops * 1e-9) / (avg_ms / 1000.0f);

        std::cout << "[Kernel 05] WMMA Tensor Cores: " << avg_ms << " ms | " << gflops << " GFLOPS\n";
    }

    std::cout << "\n=========================================================" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16); cudaFree(d_C_wmma); cudaFree(d_C_cublas_fp16);
    return 0;
}
