#ifndef WMMA_TENSOR_CORES_CUH
#define WMMA_TENSOR_CORES_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

namespace config_wmma {
    // Dimensões dos fragmentos WMMA acelerados por hardware
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Tamanho do bloco de threads (256 threads = 8 Warps)
    constexpr int BLOCK_DIM_X = 16;
    constexpr int BLOCK_DIM_Y = 16;

    // Tamanho do Tile processado por cada Bloco de GPU (128 x 128)
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 32;
}

/**
 * Kernel GEMM 05: Tensor Cores via API WMMA (Warp Matrix Multiply and Accumulate)
 * Opera em FP16 (entradas) com acumulação em FP32 para maior precisão numérica.
 */
__global__ void gemm_05_wmma_tensor_cores(
    int M, int N, int K,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C) 
{
    using namespace config_wmma;

    // Posição da Warp dentro do bloco
    const int warpId = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    const int warpsPerBlockX = (blockDim.x * blockDim.y) / 32; // 8 warps
    
    // Mapeamento das Warps para sub-blocos do resultado
    const int warpRow = warpId / 2;
    const int warpCol = warpId % 2;

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;

    // Fragmentos de registradores de hardware para a instrução mma.sync
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Inicializa o fragmento do acumulador em zero
    wmma::fill_fragment(c_frag, 0.0f);

    // Loop principal sobre a dimensão K
    for (int bk = 0; bk < K; bk += WMMA_K) {
        int aRow = blockRow * BM + warpRow * WMMA_M;
        int aCol = bk;

        int bRow = bk;
        int bCol = blockCol * BN + warpCol * WMMA_N;

        // Garante que o acesso à memória está dentro dos limites da matriz
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Carrega matriz A e B da Memória Global para os registradores da Warp
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow + bCol * K, K);

            // Executa a instrução nativa de hardware da Tensor Core (D = A * B + C)
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    // Posição final na matriz C
    int cRow = blockRow * BM + warpRow * WMMA_M;
    int cCol = blockCol * BN + warpCol * WMMA_N;

    if (cRow < M && cCol < N) {
        // Descarrega o resultado acumulado dos registradores para a Memória Global
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

#endif // WMMA_TENSOR_CORES_CUH
