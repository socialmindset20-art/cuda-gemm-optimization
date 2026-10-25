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

bool verify_results(const float* C_custom, const float* C_cublas, int M, int N) {
    float max_diff = 0.0f;
    float max_rel_diff = 0.0f;

    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float val_custom = C_custom[r * N + c];
            float val_cublas = C_cublas[r * N + c];

            float diff = std::abs(val_custom - val_cublas);
            if (diff > max_diff) {
                max_diff = diff;
            }
            float abs_ref = std::abs(val_cublas);
            if (abs_ref > 1e-3f) {
                float rel_diff = diff / abs_ref;
                if (rel_diff > max_rel_diff) {
                    max_rel_diff = rel_diff;
                }
            }
        }
    }

    std::cout << " -> Dif. Max Abs: " << max_diff << " | Dif. Max Rel: " << max_rel_diff;
    return max_rel_diff < 1e-2f;
}

int main(int argc, char** argv) {
    int M = 4096;
    int N = 4096;
    int K = 4096;

    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    std::cout << "=========================================================" << std::endl;
    std::cout << " CUDA GEMM Optimization Benchmark (C = A * B)" << std::endl;
    std::cout << " Dimensões: M = " << M << ", N = " << N << ", K = " << K << std::endl;
    std::cout << "=========================================================\n" << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_custom(M * N, 0.0f);
    std::vector<float> h_C_cublas(M * N, 0.0f);

    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand() % 10) / 10.0f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand() % 10) / 10.0f;

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

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iterations; ++i) {
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms_cublas = ms / num_iterations;
    double gflops_cublas = (total_flops * 1e-9) / (avg_ms_cublas / 1000.0f);

    CUDA_CHECK(cudaMemcpy(h_C_cublas.data(), d_C_cublas, size_C, cudaMemcpyDeviceToHost));

    std::cout << "[Reference] cuBLAS SGEMM: " << std::fixed << std::setprecision(2) 
              << avg_ms_cublas << " ms | " << gflops_cublas << " GFLOPS\n" << std::endl;

    // Kernel 00: Naive
    {
        dim3 blockDim(32, 32);
        dim3 gridDim((N + 31) / 32, (M + 31) / 32);

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iterations; ++i) {
            gemm_00_naive<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C_custom);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iterations;
        double gflops = (total_flops * 1e-9) / (avg_ms / 1000.0f);

        CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, size_C, cudaMemcpyDeviceToHost));
        std::cout << "[Kernel 00] Naive: " << avg_ms << " ms | " << gflops << " GFLOPS";
        verify_results(h_C_custom.data(), h_C_cublas.data(), M, N);
        std::cout << std::endl;
    }

    // Kernel 01: Shared Memory Tiling
    {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iterations; ++i) {
            gemm_01_shared_mem<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C_custom);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iterations;
        double gflops = (total_flops * 1e-9) / (avg_ms / 1000.0f);

        CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, size_C, cudaMemcpyDeviceToHost));
        std::cout << "[Kernel 01] Shared Mem Tiling: " << avg_ms << " ms | " << gflops << " GFLOPS";
        verify_results(h_C_custom.data(), h_C_cublas.data(), M, N);
        std::cout << std::endl;
    }

    // Kernel 02: 2D Register Tiling + float4 + Padding
    {
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iterations; ++i) {
            gemm_2d_register_tiling_float4_padded<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C_custom);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iterations;
        double gflops = (total_flops * 1e-9) / (avg_ms / 1000.0f);

        CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, size_C, cudaMemcpyDeviceToHost));
        std::cout << "[Kernel 02] 2D Reg + float4 + Pad: " << avg_ms << " ms | " << gflops << " GFLOPS";
        verify_results(h_C_custom.data(), h_C_cublas.data(), M, N);
        std::cout << std::endl;
    }

    // Kernel 03: Double Buffering
    {
        dim3 blockDim(BN / TN, BM / TM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < num_iterations; ++i) {
            gemm_double_buffering<<<gridDim, blockDim>>>(M, N, K, d_A, d_B, d_C_custom);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        float avg_ms = ms / num_iterations;
        double gflops = (total_flops * 1e-9) / (avg_ms / 1000.0f);

        CUDA_CHECK(cudaMemcpy(h_C_custom.data(), d_C_custom, size_C, cudaMemcpyDeviceToHost));
        std::cout << "[Kernel 03] Double Buffering: " << avg_ms << " ms | " << gflops << " GFLOPS";
        verify_results(h_C_custom.data(), h_C_cublas.data(), M, N);
        std::cout << std::endl;
    }

    std::cout << "\n=========================================================" << std::endl;

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_custom); cudaFree(d_C_cublas);
    return 0;
}
