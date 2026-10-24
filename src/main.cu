#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../include/gemm/naive.cuh"
#include "../include/gemm/tiled.cuh"

float measure_performance(void (*kernel)(int, int, int, float*, float*, float*), 
                          int M, int N, int K, float *d_A, float *d_B, float *d_C) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    kernel(M, N, K, d_A, d_B, d_C); // Warm-up
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    int num_runs = 10;
    for (int i = 0; i < num_runs; ++i) {
        kernel(M, N, K, d_A, d_B, d_C);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return milliseconds / num_runs;
}

int main() {
    // Reduzi um pouco o tamanho máximo para rodar mais rápido no Colab
    std::vector<int> sizes = {256, 512, 1024, 2048}; 
    
    std::cout << "Matrix_Size,Kernel,Time_ms,GFLOPS\n";

    for (int size : sizes) {
        int M = size, N = size, K = size;
        size_t bytes = M * N * sizeof(float);

        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, bytes); cudaMalloc(&d_B, bytes); cudaMalloc(&d_C, bytes);
        cudaMemset(d_A, 1, bytes); cudaMemset(d_B, 1, bytes); cudaMemset(d_C, 0, bytes);

        // 1. Testa Naive
        float time_naive = measure_performance(run_sgemm_naive, M, N, K, d_A, d_B, d_C);
        double gflops_naive = (2.0 * M * N * K) / (time_naive / 1000.0) / 1e9;
        std::cout << size << ",Naive," << time_naive << "," << gflops_naive << "\n";

        // 2. Testa Tiled
        float time_tiled = measure_performance(run_sgemm_tiled, M, N, K, d_A, d_B, d_C);
        double gflops_tiled = (2.0 * M * N * K) / (time_tiled / 1000.0) / 1e9;
        std::cout << size << ",Tiled," << time_tiled << "," << gflops_tiled << "\n";

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
    return 0;
}
