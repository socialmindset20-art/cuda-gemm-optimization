#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "kernels/00_naive.cuh"
#include "kernels/01_shared_mem.cuh"
#include "kernels/02_reg_tiling.cuh"
#include "kernels/03_double_buffering.cuh"
#include "kernels/04_fp16_simd.cuh"
#include "kernels/05_wmma_tensor_op.cuh"
#include "kernels/06_generic_gemm.cuh"
#include "kernels/07_cutlass_basic.cuh"
#include "kernels/08_cutlass_tensor_op.cuh"
#include "kernels/09_hopper_ampere_sol.cuh"

void checkCudaStatus(cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(status) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    int M = 4096, N = 4096, K = 4096;
    if (argc >= 4) {
        M = std::atoi(argv[1]);
        N = std::atoi(argv[2]);
        K = std::atoi(argv[3]);
    }

    std::cout << "=========================================================" << std::endl;
    std::cout << " CUDA GEMM Optimization Benchmark (Kernel 00 -> 09 SOL)" << std::endl;
    std::cout << " Dimensões: M = " << M << ", N = " << N << ", K = " << K << std::endl;
    std::cout << "=========================================================" << std::endl << std::endl;

    double flops = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);

    size_t bytes_fp32_A = M * K * sizeof(float);
    size_t bytes_fp32_B = K * N * sizeof(float);
    size_t bytes_fp32_C = M * N * sizeof(float);

    size_t bytes_fp16_A = M * K * sizeof(__half);
    size_t bytes_fp16_B = K * N * sizeof(__half);

    float *h_A = (float*)malloc(bytes_fp32_A);
    float *h_B = (float*)malloc(bytes_fp32_B);
    float *h_C_ref = (float*)malloc(bytes_fp32_C);
    float *h_C_out = (float*)malloc(bytes_fp32_C);

    __half *h_A_fp16 = (__half*)malloc(bytes_fp16_A);
    __half *h_B_fp16 = (__half*)malloc(bytes_fp16_B);

    for (int i = 0; i < M * K; ++i) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_A[i] = val;
        h_A_fp16[i] = __float2half(val);
    }
    for (int i = 0; i < K * N; ++i) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = val;
        h_B_fp16[i] = __float2half(val);
    }

    float *d_A, *d_B, *d_C;
    __half *d_A_fp16, *d_B_fp16;

    checkCudaStatus(cudaMalloc(&d_A, bytes_fp32_A), "d_A malloc");
    checkCudaStatus(cudaMalloc(&d_B, bytes_fp32_B), "d_B malloc");
    checkCudaStatus(cudaMalloc(&d_C, bytes_fp32_C), "d_C malloc");

    checkCudaStatus(cudaMalloc(&d_A_fp16, bytes_fp16_A), "d_A_fp16 malloc");
    checkCudaStatus(cudaMalloc(&d_B_fp16, bytes_fp16_B), "d_B_fp16 malloc");

    checkCudaStatus(cudaMemcpy(d_A, h_A, bytes_fp32_A, cudaMemcpyHostToDevice), "A H2D");
    checkCudaStatus(cudaMemcpy(d_B, h_B, bytes_fp32_B, cudaMemcpyHostToDevice), "B H2D");

    checkCudaStatus(cudaMemcpy(d_A_fp16, h_A_fp16, bytes_fp16_A, cudaMemcpyHostToDevice), "A_fp16 H2D");
    checkCudaStatus(cudaMemcpy(d_B_fp16, h_B_fp16, bytes_fp16_B, cudaMemcpyHostToDevice), "B_fp16 H2D");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;

    // Reference: cuBLAS SGEMM
    cudaEventRecord(start);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_cublas = 0;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    std::cout << "[Reference] cuBLAS SGEMM (FP32): " << ms_cublas << " ms | " << (flops * 1e-6 / ms_cublas) << " GFLOPS" << std::endl;

    // Reference: cuBLAS HGEMM
    cudaEventRecord(start);
    cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B_fp16, CUDA_R_16F, N, d_A_fp16, CUDA_R_16F, K, &beta, d_C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_cublas_hgemm = 0;
    cudaEventElapsedTime(&ms_cublas_hgemm, start, stop);
    std::cout << "[Reference] cuBLAS HGEMM (FP16): " << ms_cublas_hgemm << " ms | " << (flops * 1e-6 / ms_cublas_hgemm) << " GFLOPS" << std::endl;

    // Kernel 07: CUTLASS Basic GEMM
    cudaMemset(d_C, 0, bytes_fp32_C);
    cudaEventRecord(start);
    run_07_cutlass_basic(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_k07 = 0;
    cudaEventElapsedTime(&ms_k07, start, stop);
    std::cout << "[Kernel 07] CUTLASS Basic GEMM: " << ms_k07 << " ms | " << (flops * 1e-6 / ms_k07) << " GFLOPS" << std::endl;

    // Kernel 08: CUTLASS TensorOp
    cudaMemset(d_C, 0, bytes_fp32_C);
    cudaEventRecord(start);
    run_08_cutlass_tensor_op(M, N, K, alpha, d_A_fp16, d_B_fp16, beta, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_k08 = 0;
    cudaEventElapsedTime(&ms_k08, start, stop);
    std::cout << "[Kernel 08] CUTLASS TensorOp (FP16): " << ms_k08 << " ms | " << (flops * 1e-6 / ms_k08) << " GFLOPS" << std::endl;

    // Kernel 09: Hopper/Ampere Speed-Of-Light PTX
    cudaMemset(d_C, 0, bytes_fp32_C);
    dim3 grid_sol((N + 127) / 128, (M + 127) / 128);
    dim3 block_sol(16, 16);
    cudaEventRecord(start);
    gemm_09_hopper_ampere_sol<<<grid_sol, block_sol>>>(M, N, K, alpha, d_A_fp16, d_B_fp16, beta, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_k09 = 0;
    cudaEventElapsedTime(&ms_k09, start, stop);
    std::cout << "[Kernel 09] Hopper/Ampere SOL PTX (FP16): " << ms_k09 << " ms | " << (flops * 1e-6 / ms_k09) << " GFLOPS" << std::endl;

    std::cout << "=========================================================" << std::endl;

    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_A_fp16); cudaFree(d_B_fp16);
    free(h_A); free(h_B); free(h_C_ref); free(h_C_out);
    free(h_A_fp16); free(h_B_fp16);

    return 0;
}
