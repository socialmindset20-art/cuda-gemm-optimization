#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "kernels/00_naive.cuh"
#include "kernels/01_shared_mem.cuh"
#include "kernels/02_float4_padded.cuh"
#include "kernels/03_double_buffering.cuh"
#include "kernels/04_fp16_half2.cuh"
#include "kernels/05_wmma_tensor_cores.cuh"
#include "kernels/06_generic_gemm.cuh"

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
    // Dimensões não-quadradas e não-múltiplas de 128 por padrão
    int M = 1000, N = 2000, K = 1500;
    if (argc >= 4) {
        M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]);
    }

    float alpha = 2.0f;
    float beta = 1.0f;

    std::cout << "=========================================================" << std::endl;
    std::cout << " CUDA Generic GEMM Benchmark (C = alpha * A*B + beta * C)" << std::endl;
    std::cout << " Dimensões: M = " << M << ", N = " << N << ", K = " << K << std::endl;
    std::cout << " Parâmetros: alpha = " << alpha << ", beta = " << beta << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_init(M * N);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand() % 10) / 10.0f;
    for (int i = 0; i < M * N; ++i) h_C_init[i] = static_cast<float>(rand() % 10) / 10.0f;

    float *d_A, *d_B, *d_C_custom, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C_custom, size_C));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, size_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    double total_flops = 2.0 * static_cast<double>(M) * N * K;
    const int num_iterations = 5;
    float ms = 0.0f;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // --- Reference: cuBLAS SGEMM com alpha e beta ---
    CUDA_CHECK(cudaMemcpy(d_C_cublas, h_C_init.data(), size_C, cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                                &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms_cublas = ms / num_iterations;
    double gflops_cublas = (total_flops * 1e-9) / (avg_ms_cublas / 1000.0f);

    std::cout << "[Reference] cuBLAS SGEMM: " << std::fixed << std::setprecision(2) 
              << avg_ms_cublas << " ms | " << gflops_cublas << " GFLOPS\n" << std::endl;

    // --- Kernel 06: Generic GEMM ---
    CUDA_CHECK(cudaMemcpy(d_C_custom, h_C_init.data(), size_C, cudaMemcpyHostToDevice));

    dim3 blockDim(GENERIC_BN / GENERIC_TN, GENERIC_BM / GENERIC_TM);
    dim3 gridDim((N + GENERIC_BN - 1) / GENERIC_BN, (M + GENERIC_BM - 1) / GENERIC_BM);

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        gemm_06_generic<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C_custom);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / num_iterations;
    double gflops = (total_flops * 1e-9) / (avg_ms / 1000.0f);

    // Validação Numérica
    std::vector<float> h_C_custom(M * N);
    std::vector<float> h_C_cublas(M * N);
    CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, size_C, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, size_C, cudaMemcpyDeviceToHost));

    float max_diff = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::abs(h_C_custom[i] - h_C_cublas[i]);
        if (diff > max_diff) max_diff = diff;
    }

    std::cout << "[Kernel 06] Generic GEMM: " << avg_ms << " ms | " << gflops << " GFLOPS"
              << " -> Max Diff vs cuBLAS: " << std::setprecision(4) << max_diff << "\n";

    std::cout << "\n=========================================================" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_custom); cudaFree(d_C_cublas);
    return 0;
}
