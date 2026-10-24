#include "../../include/gemm/tiled.cuh"
#include <math.h>

#define TILE_SIZE 16

__global__ void sgemm_tiled_kernel(int M, int N, int K, float *A, float *B, float *C) {
    // Aloca memória compartilhada (muito mais rápida que a global)
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop sobre os blocos (tiles) da matriz
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int ph = 0; ph < num_tiles; ++ph) {
        
        // Carrega os dados da memória global para a compartilhada de forma colaborativa
        if (row < M && (ph * TILE_SIZE + tx) < K)
            sA[ty][tx] = A[row * K + ph * TILE_SIZE + tx];
        else
            sA[ty][tx] = 0.0f;

        if ((ph * TILE_SIZE + ty) < K && col < N)
            sB[ty][tx] = B[(ph * TILE_SIZE + ty) * N + col];
        else
            sB[ty][tx] = 0.0f;

        // Espera todas as threads do bloco terminarem de carregar
        __syncthreads();

        // Multiplica usando os dados rápidos da memória compartilhada
        for (int i = 0; i < TILE_SIZE; ++i) {
            sum += sA[ty][i] * sB[i][tx];
        }
        
        // Espera terminar a conta antes de carregar o próximo bloco
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void run_sgemm_tiled(int M, int N, int K, float *A, float *B, float *C) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    
    sgemm_tiled_kernel<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
