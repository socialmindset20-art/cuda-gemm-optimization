#ifndef SHARED_MEM_CUH
#define SHARED_MEM_CUH

#include <cuda_runtime.h>

constexpr int TILE_SIZE = 32;

/**
 * Kernel GEMM 01: Shared Memory Tiling
 * C = A * B
 * 
 * Divide as matrizes em blocos de 32x32 para reuso local na Shared Memory.
 */
__global__ void gemm_01_shared_mem(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) 
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int bk = 0; bk < K; bk += TILE_SIZE) {

        if (row < M && (bk + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + (bk + threadIdx.x)];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ((bk + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(bk + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

#endif // SHARED_MEM_CUH
