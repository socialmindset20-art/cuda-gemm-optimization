#ifndef FLOAT4_PADDED_CUH
#define FLOAT4_PADDED_CUH

#include <cuda_runtime.h>

constexpr int BM = 128; 
constexpr int BN = 128; 
constexpr int BK = 16;  
constexpr int TM = 8;   
constexpr int TN = 8;   
constexpr int PADDING = 4; 

__global__ void gemm_2d_register_tiling_float4_padded(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) 
{
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    __shared__ float As[BM][BK + PADDING];
    __shared__ float Bs[BK][BN + PADDING];

    float threadResults[TM][TN] = {0.0f};
    float regA[TM] = {0.0f};
    float regB[TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int aSharedRow = tid / (BK / 4);
    const int aSharedCol = (tid % (BK / 4)) * 4;

    const int bSharedRow = tid / (BN / 4);
    const int bSharedCol = (tid % (BN / 4)) * 4;

    for (int bk = 0; bk < K; bk += BK) {
        
        #pragma unroll
        for (int loadOffset = 0; loadOffset < BM; loadOffset += (256 / (BK / 4))) {
            int currRow = aSharedRow + loadOffset;
            int aGlobalRow = blockRow * BM + currRow;
            int aGlobalCol = bk + aSharedCol;

            if (aGlobalRow < M && aGlobalCol < K) {
                float4 tmpA = reinterpret_cast<const float4*>(&A[aGlobalRow * K + aGlobalCol])[0];
                As[currRow][aSharedCol + 0] = tmpA.x;
                As[currRow][aSharedCol + 1] = tmpA.y;
                As[currRow][aSharedCol + 2] = tmpA.z;
                As[currRow][aSharedCol + 3] = tmpA.w;
            } else {
                As[currRow][aSharedCol + 0] = 0.0f;
                As[currRow][aSharedCol + 1] = 0.0f;
                As[currRow][aSharedCol + 2] = 0.0f;
                As[currRow][aSharedCol + 3] = 0.0f;
            }
        }

        #pragma unroll
        for (int loadOffset = 0; loadOffset < BK; loadOffset += (256 / (BN / 4))) {
            int currRow = bSharedRow + loadOffset;
            int bGlobalRow = bk + currRow;
            int bGlobalCol = blockCol * BN + bSharedCol;

            if (bGlobalRow < K && bGlobalCol < N) {
                float4 tmpB = reinterpret_cast<const float4*>(&B[bGlobalRow * N + bGlobalCol])[0];
                Bs[currRow][bSharedCol + 0] = tmpB.x;
                Bs[currRow][bSharedCol + 1] = tmpB.y;
                Bs[currRow][bSharedCol + 2] = tmpB.z;
                Bs[currRow][bSharedCol + 3] = tmpB.w;
            } else {
                Bs[currRow][bSharedCol + 0] = 0.0f;
                Bs[currRow][bSharedCol + 1] = 0.0f;
                Bs[currRow][bSharedCol + 2] = 0.0f;
                Bs[currRow][bSharedCol + 3] = 0.0f;
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[threadRow * TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[k][threadCol * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; j += 4) {
            int globalRow = blockRow * BM + threadRow * TM + i;
            int globalCol = blockCol * BN + threadCol * TN + j;

            if (globalRow < M && globalCol < N) {
                float4 tmpC;
                tmpC.x = threadResults[i][j + 0];
                tmpC.y = threadResults[i][j + 1];
                tmpC.z = threadResults[i][j + 2];
                tmpC.w = threadResults[i][j + 3];

                reinterpret_cast<float4*>(&C[globalRow * N + globalCol])[0] = tmpC;
            }
        }
    }
}

#endif // FLOAT4_PADDED_CUH
