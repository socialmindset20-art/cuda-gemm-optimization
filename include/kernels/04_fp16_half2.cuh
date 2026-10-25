#ifndef FP16_HALF2_CUH
#define FP16_HALF2_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>

constexpr int FP16_BM = 128; 
constexpr int FP16_BN = 128; 
constexpr int FP16_BK = 16;  // 16 elementos half2 = 32 valores half
constexpr int FP16_TM = 8;   
constexpr int FP16_TN = 8;   

/**
 * Kernel GEMM 04: FP16 utilizando instrução intrínseca __hfma2
 */
__global__ void gemm_04_fp16_half2(
    int M, int N, int K,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C) 
{
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    __shared__ half2 As[FP16_BM][FP16_BK];
    __shared__ half2 Bs[FP16_BK][FP16_BN / 2];

    half2 threadResults[FP16_TM][FP16_TN / 2];
    for (int i = 0; i < FP16_TM; ++i) {
        for (int j = 0; j < FP16_TN / 2; ++j) {
            threadResults[i][j] = __float2half2_rn(0.0f);
        }
    }

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const half2* A_half2 = reinterpret_cast<const half2*>(A);
    const half2* B_half2 = reinterpret_cast<const half2*>(B);

    int K_half2 = K / 2;
    int N_half2 = N / 2;

    for (int bk = 0; bk < K_half2; bk += FP16_BK) {
        int aRow = tid / FP16_BK;
        int aCol = tid % FP16_BK;
        if ((blockRow * FP16_BM + aRow) < M && (bk + aCol) < K_half2) {
            As[aRow][aCol] = A_half2[(blockRow * FP16_BM + aRow) * K_half2 + (bk + aCol)];
        } else {
            As[aRow][aCol] = __float2half2_rn(0.0f);
        }

        int bRow = tid / (FP16_BN / 2);
        int bCol = tid % (FP16_BN / 2);
        if ((bk + bRow) < K_half2 && (blockCol * (FP16_BN / 2) + bCol) < N_half2) {
            Bs[bRow][bCol] = B_half2[(bk + bRow) * N_half2 + (blockCol * (FP16_BN / 2) + bCol)];
        } else {
            Bs[bRow][bCol] = __float2half2_rn(0.0f);
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < FP16_BK; ++k) {
            #pragma unroll
            for (int i = 0; i < FP16_TM; ++i) {
                half2 regA = As[threadRow * FP16_TM + i][k];
                #pragma unroll
                for (int j = 0; j < FP16_TN / 2; ++j) {
                    half2 regB = Bs[k][threadCol * (FP16_TN / 2) + j];
                    threadResults[i][j] = __hfma2(regA, regB, threadResults[i][j]);
                }
            }
        }

        __syncthreads();
    }

    half2* C_half2 = reinterpret_cast<half2*>(C);
    #pragma unroll
    for (int i = 0; i < FP16_TM; ++i) {
        #pragma unroll
        for (int j = 0; j < FP16_TN / 2; ++j) {
            int globalRow = blockRow * FP16_BM + threadRow * FP16_TM + i;
            int globalCol = blockCol * N_half2 + threadCol * (FP16_TN / 2) + j;

            if (globalRow < M && globalCol < N_half2) {
                C_half2[globalRow * N_half2 + globalCol] = threadResults[i][j];
            }
        }
    }
}

#endif // FP16_HALF2_CUH
