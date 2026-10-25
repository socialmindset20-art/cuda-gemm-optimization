#ifndef DOUBLE_BUFFERING_CUH
#define DOUBLE_BUFFERING_CUH

#include <cuda_runtime.h>

constexpr int BM = 128; 
constexpr int BN = 128; 
constexpr int BK = 16;  
constexpr int TM = 8;   
constexpr int TN = 8;   
constexpr int PADDING = 4; 

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

    __shared__ float As[2][BM][BK + PADDING];
    __shared__ float Bs[2][BK][BN + PADDING];

    float threadResults[TM][TN] = {0.0f};
    float regA[TM] = {0.0f};
    float regB[TN] = {0.0f};

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int aSharedRow = tid / (BK / 4);
    const int aSharedCol = (tid % (BK / 4)) * 4;

    const int bSharedRow = tid / (BN / 4);
    const int bSharedCol = (tid % (BN / 4)) * 4;

    int writeStage = 0;
    int readStage = 0;

    int bk = 0;
    
    // Prolog
    #pragma unroll
    for (int loadOffset = 0; loadOffset < BM; loadOffset += (256 / (BK / 4))) {
        int currRow = aSharedRow + loadOffset;
        int aGlobalRow = blockRow * BM + currRow;
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
    for (int loadOffset = 0; loadOffset < BK; loadOffset += (256 / (BN / 4))) {
        int currRow = bSharedRow + loadOffset;
        int bGlobalRow = bk + currRow;
        int bGlobalCol = blockCol * BN + bSharedCol;

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
    for (bk = BK; bk < K; bk += BK) {
        writeStage ^= 1;

        #pragma unroll
        for (int loadOffset = 0; loadOffset < BM; loadOffset += (256 / (BK / 4))) {
            int currRow = aSharedRow + loadOffset;
            int aGlobalRow = blockRow * BM + currRow;
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
        for (int loadOffset = 0; loadOffset < BK; loadOffset += (256 / (BN / 4))) {
            int currRow = bSharedRow + loadOffset;
            int bGlobalRow = bk + currRow;
            int bGlobalCol = blockCol * BN + bSharedCol;

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
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[readStage][threadRow * TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[readStage][k][threadCol * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults
