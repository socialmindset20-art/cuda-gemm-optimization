#ifndef WMMA_TENSOR_CORES_CUH
#define WMMA_TENSOR_CORES_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// Dimensões dos fragmentos WMMA acelerados por hardware
constexpr int WMMA_M_FRAG = 16;
constexpr int WMMA_N_FRAG = 16;
constexpr int WMMA_K_FRAG = 16;

// Dimensões dos blocos e tiles do kernel
constexpr int WMMA_BLOCK_DIM_X = 16;
constexpr int WMMA_BLOCK_DIM_Y = 16;

constexpr int WMMA_BM = 128;
constexpr int WMMA_BN = 128;
constexpr int WMMA_BK = 32;

/**
 * Kernel GEMM 05: Tensor Cores via API WMMA (Warp Matrix Multiply and Accumulate)
 */
__global__ void gemm_05_wmma_tensor_cores(
    int M, int N, int K,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C) 
{
    // Posição da Warp dentro do bloco
    const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    
    // Mapeamento das Warps para sub-blocos
    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Fragmentos de registradores de hardware para a instrução mma.sync
    wmma::fragment<wmma::matrix_a, WMMA_M_FRAG, WMMA_N_FRAG, WMMA_K_FRAG, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M_FRAG, WMMA_N_FRAG, WMMA_K_FRAG, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M_FRAG, WMMA_N_FRAG, WMMA_K_FRAG, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Loop principal sobre a dimensão K
    for (int bk = 0; bk < K; bk += WMMA_K_FRAG) {
        int aRow = blockRow * WMMA_BM + warpRow * WMMA_M_FRAG;
        int aCol = bk;

        int bRow = bk;
        int bCol = blockCol * WMMA_BN + warpCol * WMMA_N_FRAG;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow + bCol * K, K);

            // Instrução nativa de hardware na Tensor Core
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = blockRow * WMMA_BM + warpRow * WMMA_M_FRAG;
    int cCol = blockCol * WMMA_BN + warpCol * WMMA_N_FRAG;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

#endif // WMMA_TENSOR_CORES_CUH
