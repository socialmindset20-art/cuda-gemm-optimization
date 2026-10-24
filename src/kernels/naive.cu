#include "../../include/gemm/naive.cuh"

__global__ void sgemm_naive_kernel(int M, int N, int K, float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void run_sgemm_naive(int M, int N, int K, float *A, float *B, float *C) {
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
    
    sgemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
