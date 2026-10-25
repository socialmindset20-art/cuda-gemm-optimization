#ifndef NAIVE_CUH
#define NAIVE_CUH

#include <cuda_runtime.h>

/**
 * Kernel GEMM 00: Naive / Baseline
 * C = A * B
 */
__global__ void gemm_00_naive(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) 
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

#endif // NAIVE_CUH
