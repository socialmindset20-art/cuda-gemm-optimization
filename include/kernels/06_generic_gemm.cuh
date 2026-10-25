#ifndef GENERIC_GEMM_CUH
#define GENERIC_GEMM_CUH

#include <cuda_runtime.h>

constexpr int GENERIC_BM = 64;
constexpr int GENERIC_BN = 64;
constexpr int GENERIC_BK = 16;
constexpr int GENERIC_TM = 4;
constexpr int GENERIC_TN = 4;

/**
 * Kernel GEMM 06: Suporta dimensões arbitrárias (M, N, K) e a fórmula C = alpha * (A * B) + beta * C
 */
__global__ void gemm_06_generic(
    int M, int N, int K,
    float alpha,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float beta,
    float* __restrict__ C) 
{
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    __shared__ float As[GENERIC_BM][GENERIC_BK];
    __shared__ float Bs[GENERIC_BK][GENERIC_BN];

    float threadResults[GENERIC_TM][GENERIC_TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int numThreads = blockDim.x * blockDim.y;

    for (int bk = 0; bk < K; bk += GENERIC_BK) {

        // 1. Carregamento de A -> Shared Memory com Boundary Check
        for (int loadOffset = tid; loadOffset < GENERIC_BM * GENERIC_BK; loadOffset += numThreads) {
            int aRow = loadOffset / GENERIC_BK;
            int aCol = loadOffset % GENERIC_BK;

            int globalRow = blockRow * GENERIC_BM + aRow;
            int globalCol = bk + aCol;

            if (globalRow < M && globalCol < K) {
                As[aRow][aCol] = A[globalRow * K + globalCol];
            } else {
                As[aRow][aCol] = 0.0f;
            }
        }

        // 2. Carregamento de B -> Shared Memory com Boundary Check
        for (int loadOffset = tid; loadOffset < GENERIC_BK * GENERIC_BN; loadOffset += numThreads) {
            int bRow = loadOffset / GENERIC_BN;
            int bCol = loadOffset % GENERIC_BN;

            int globalRow = bk + bRow;
            int globalCol = blockCol * GENERIC_BN + bCol;

            if (globalRow < K && globalCol < N) {
                Bs[bRow][bCol] = B[globalRow * N + globalCol];
            } else {
                Bs[bRow][bCol] = 0.0f;
            }
        }

        __syncthreads();

        // 3. Multiplicação
        #pragma unroll
        for (int k = 0; k < GENERIC_BK; ++k) {
            #pragma unroll
            for (int i = 0; i < GENERIC_TM; ++i) {
                float regA = As[threadRow * GENERIC_TM + i][k];
                #pragma unroll
                for (int j = 0; j < GENERIC_TN; ++j) {
                    threadResults[i][j] += regA * Bs[k][threadCol * GENERIC_TN + j];
                }
            }
        }

        __syncthreads();
    }

    // 4. Escrita em C com aplicação de alpha e beta
    #pragma unroll
    for (int i = 0; i < GENERIC_TM; ++i) {
        #pragma unroll
        for (int j = 0; j < GENERIC_TN; ++j) {
            int globalRow = blockRow * GENERIC_BM + threadRow * GENERIC_TM + i;
            int globalCol = blockCol * GENERIC_BN + threadCol * GENERIC_TN + j;

            if (globalRow < M && globalCol < N) {
                int cIdx = globalRow * N + globalCol;

                if (beta == 0.0f) {
                    C[cIdx] = alpha * threadResults[i][j];
                } else {
                    C[cIdx] = alpha * threadResults[i][j] + beta * C[cIdx];
                }
            }
        }
    }
}

#endif // GENERIC_GEMM_CUH
