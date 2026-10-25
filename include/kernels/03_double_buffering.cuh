#ifndef DOUBLE_BUFFERING_CUH
#define DOUBLE_BUFFERING_CUH

#include <cuda_runtime.h>

constexpr int DB_BM = 128; 
constexpr int DB_BN = 128; 
constexpr int DB_BK = 16;  
constexpr int DB_TM = 8;   
constexpr int DB_TN = 8;   
constexpr int DB_PADDING = 4; 

__global__ void gemm_double_buffering(
    int M, int N, int K,
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C) 
{
    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    const int threadRow = threadIdx.y;
    const int threadCol = threadIdx.x;

    __shared__ float As[2][DB_BM][DB_BK + DB_PADDING];
    __shared__ float Bs[2][DB_BK][DB_BN + DB_PADDING];

    float threadResults[DB_TM][DB_TN] = {0.0f};
    float regA[DB_TM] = {0.0f};
    float regB[DB_TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int aSharedRow = tid / (DB_BK / 4);
    const int aSharedCol = (tid % (DB_BK / 4)) * 4;

    const int bSharedRow = tid / (DB_BN / 4);
    const int bSharedCol = (tid % (DB_BN / 4)) * 4;

    int writeStage = 0;
    int readStage = 0;

    int bk = 0;
    
    // Prolog
    #pragma unroll
    for (int loadOffset = 0; loadOffset < DB_BM; loadOffset += (256 / (DB_BK / 4))) {
        int currRow = aSharedRow + loadOffset;
        int aGlobalRow = blockRow * DB_BM + currRow;
        int aGlobalCol = bk + aSharedCol;

        if (aGlobalRow < M && aGlobalCol < K) {
            float4 tmpA = reinterpret_cast<const float4*>(&A[aGlobalRow * K + aGlobalCol])[0];
            As[writeStage][currRow][aSharedCol + 0] = tmpA.x;
            As[writeStage][currRow][aSharedCol + 1] = tmpA.y;
            As[writeStage][currRow][aSharedCol + 2] = tmpA.z;
            As[writeStage][currRow][aSharedCol + 3] = tmpA.w;
        } else {
            As[writeStage][currRow][aSharedCol + 0] = 0.0f;
            As[writeStage][currRow][aSharedCol + 1] = 0.0f;
            As[writeStage][currRow][aSharedCol + 2] = 0.0f;
            As[writeStage][currRow][aSharedCol + 3] = 0.0f;
        }
    }

    #pragma unroll
    for (int loadOffset = 0; loadOffset < DB_BK; loadOffset += (256 / (DB_BN / 4))) {
        int currRow = bSharedRow + loadOffset;
        int bGlobalRow = bk + currRow;
        int bGlobalCol = blockCol * DB_BN + bSharedCol;

        if (bGlobalRow < K && bGlobalCol < N) {
            float4 tmpB = reinterpret_cast<const float4*>(&B[bGlobalRow * N + bGlobalCol])[0];
            Bs[writeStage][currRow][bSharedCol + 0] = tmpB.x;
            Bs[writeStage][currRow][bSharedCol + 1] = tmpB.y;
            Bs[writeStage][currRow][bSharedCol + 2] = tmpB.z;
            Bs[writeStage][currRow][bSharedCol + 3] = tmpB.w;
        } else {
            Bs[writeStage][currRow][bSharedCol + 0] = 0.0f;
            Bs[writeStage][currRow][bSharedCol + 1] = 0.0f;
            Bs[writeStage][currRow][bSharedCol + 2] = 0.0f;
            Bs[writeStage][currRow][bSharedCol + 3] = 0.0f;
        }
    }

    __syncthreads();

    // Main Loop
    for (bk = DB_BK; bk < K; bk += DB_BK) {
        writeStage ^= 1;

        #pragma unroll
        for (int loadOffset = 0; loadOffset < DB_BM; loadOffset += (256 / (DB_BK / 4))) {
            int currRow = aSharedRow + loadOffset;
            int aGlobalRow = blockRow * DB_BM + currRow;
            int aGlobalCol = bk + aSharedCol;

            if (aGlobalRow < M && aGlobalCol < K) {
                float4 tmpA = reinterpret_cast<const float4*>(&A[aGlobalRow * K + aGlobalCol])[0];
                As[writeStage][currRow][aSharedCol + 0] = tmpA.x;
                As[writeStage][currRow][aSharedCol + 1] = tmpA.y;
                As[writeStage][currRow][aSharedCol + 2] = tmpA.z;
                As[writeStage][currRow][aSharedCol + 3] = tmpA.w;
            } else {
                As[writeStage][currRow][aSharedCol + 0] = 0.0f;
                As[writeStage][currRow][aSharedCol + 1] = 0.0f;
                As[writeStage][currRow][aSharedCol + 2] = 0.0f;
                As[writeStage][currRow][aSharedCol + 3] = 0.0f;
            }
        }

        #pragma unroll
        for (int loadOffset = 0; loadOffset < DB_BK; loadOffset += (256 / (DB_BN / 4))) {
            int currRow = bSharedRow + loadOffset;
            int bGlobalRow = bk + currRow;
            int bGlobalCol = blockCol * DB_BN + bSharedCol;

            if (bGlobalRow < K && bGlobalCol < N) {
                float4 tmpB = reinterpret_cast<const float4*>(&B[bGlobalRow * N + bGlobalCol])[0];
                Bs[writeStage][currRow][bSharedCol + 0] = tmpB.x;
                Bs[writeStage][currRow][bSharedCol + 1] = tmpB.y;
                Bs[writeStage][currRow][bSharedCol + 2] = tmpB.z;
                Bs[writeStage][currRow][bSharedCol + 3] = tmpB.w;
            } else {
                Bs[writeStage][currRow][bSharedCol + 0] = 0.0f;
                Bs[writeStage][currRow][bSharedCol + 1] = 0.0f;
                Bs[writeStage][currRow][bSharedCol + 2] = 0.0f;
                Bs[writeStage][currRow][bSharedCol + 3] = 0.0f;
            }
        }

        #pragma unroll
        for (int k = 0; k < DB_BK; ++k) {
            #pragma unroll
            for (int i = 0; i < DB_TM; ++i) {
                regA[i] = As[readStage][threadRow * DB_TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < DB_TN; ++j) {
                regB[j] = Bs[readStage][k][threadCol * DB_TN + j];
            }

            #pragma unroll
            for (int i = 0; i < DB_TM; ++i) {
                #pragma unroll
                for (int j = 0; j < DB_TN; ++j) {
                    threadResults[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
        readStage ^= 1;
    }

    // Epilog
    #pragma unroll
    for (int k = 0; k < DB_BK; ++k) {
        #pragma unroll
        for (int i = 0; i < DB_TM; ++i) {
            regA[i] = As[readStage][threadRow * DB_TM + i][k];
        }

        #pragma unroll
        for (int j = 0; j < DB_TN; ++j) {
            regB[j] = Bs[readStage][k][threadCol * DB_TN + j];
        }

        #pragma unroll
        for (int i = 0; i < DB_TM; ++i) {
            #pragma unroll
            for (int j = 0; j < DB_TN; ++j) {
                threadResults[i][j] += regA[i] * regB[j];
            }
        }
    }

    // Write back
    #pragma unroll
    for (int i = 0; i < DB_TM; ++i) {
        #pragma unroll
        for (int j = 0; j < DB_TN; j += 4) {
            int globalRow = blockRow * DB_BM + threadRow * DB_TM + i;
            int globalCol = blockCol * DB_BN + threadCol * DB_TN + j;

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

#endif // DOUBLE_BUFFERING_CUH
