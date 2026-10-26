#ifndef HOPPER_AMPERE_SOL_CUH
#define HOPPER_AMPERE_SOL_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

constexpr int SOL_BM = 128;
constexpr int SOL_BN = 128;
constexpr int SOL_BK = 32;
constexpr int SOL_PAD_K = 8;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

// --- IMPLEMENTAÇÃO NATIVA PTX PARA AMPERE/HOPPER (sm_80+) ---

__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* gmem_ptr) {
    uint32_t smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_addr), "l"(gmem_ptr)
    );
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k16(
    float* c_accum, 
    const unsigned int* a_reg, 
    const unsigned int* b_reg) 
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};\n"
        : "+f"(c_accum[0]), "+f"(c_accum[1]), "+f"(c_accum[2]), "+f"(c_accum[3])
        : "r"(a_reg[0]), "r"(a_reg[1]),
          "r"(b_reg[0])
    );
}

__global__ void gemm_09_hopper_ampere_sol(
    int M, int N, int K,
    float alpha,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    __shared__ __align__(16) __half smem_A[2][SOL_BM][SOL_BK + SOL_PAD_K];
    __shared__ __align__(16) __half smem_B[2][SOL_BK][SOL_BN + SOL_PAD_K];

    const int blockRow = blockIdx.y;
    const int blockCol = blockIdx.x;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    const int load_a_row = tid / (SOL_BK / 8);
    const int load_a_col = (tid % (SOL_BK / 8)) * 8;
    const int load_b_row = tid / (SOL_BN / 8);
    const int load_b_col = (tid % (SOL_BN / 8)) * 8;

    const int global_a_row = blockRow * SOL_BM + load_a_row;
    const int global_b_col = blockCol * SOL_BN + load_b_col;

    float accum[4][8] = {0.0f};
    int write_stage = 0;

    if (global_a_row < M && load_a_col < K) {
        cp_async_cg(&smem_A[write_stage][load_a_row][load_a_col], &A[global_a_row * K + load_a_col]);
    }
    if (load_b_row < K && global_b_col < N) {
        cp_async_cg(&smem_B[write_stage][load_b_row][load_b_col], &B[load_b_row * N + global_b_col]);
    }
    
    cp_async_commit();
    cp_async_wait_group<0>();
    __syncthreads();

    for (int k_tile = SOL_BK; k_tile < K; k_tile += SOL_BK) {
        int read_stage = write_stage;
        write_stage ^= 1;

        int global_a_col = k_tile + load_a_col;
        int global_b_row = k_tile + load_b_row;

        if (global_a_row < M && global_a_col < K) {
            cp_async_cg(&smem_A[write_stage][load_a_row][load_a_col], &A[global_a_row * K + global_a_col]);
        }
        if (global_b_row < K && global_b_col < N) {
            cp_async_cg(&smem_B[write_stage][load_b_row][load_b_col], &B[load_b_row * N + global_b_col]);
        }
        cp_async_commit();

        #pragma unroll
        for (int k_step = 0; k_step < SOL_BK; k_step += 16) {
            unsigned int a_reg[2];
            unsigned int b_reg[1];

            a_reg[0] = *reinterpret_cast<unsigned int*>(&smem_A[read_stage][threadIdx.y * 16][k_step]);
            a_reg[1] = *reinterpret_cast<unsigned int*>(&smem_A[read_stage][threadIdx.y * 16 + 8][k_step]);
            b_reg[0] = *reinterpret_cast<unsigned int*>(&smem_B[read_stage][k_step][threadIdx.x * 8]);

            mma_m16n8k16(accum[0], a_reg, b_reg);
        }

        cp_async_wait_group<0>();
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int row = blockRow * SOL_BM + threadIdx.y * 16 + i * 4;
            int col = blockCol * SOL_BN + threadIdx.x * 8 + j;
            if (row < M && col < N) {
                int c_idx = row * N + col;
                if (beta == 0.0f) {
                    C[c_idx] = alpha * accum[i][j];
                } else {
                    C[c_idx] = alpha * accum[i][j] + beta * C[c_idx];
                }
            }
        }
    }
}

#else

// --- FALLBACK COMPATÍVEL COM TURING / VOLTA (sm_70 / sm_75) VIA WMMA ---

using namespace nvcuda;

__global__ void gemm_09_hopper_ampere_sol(
    int M, int N, int K,
    float alpha,
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float beta,
    float* __restrict__ C)
{
    const int WARP_SIZE = 32;
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            wmma::load_matrix_sync(a_frag, A + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, B + bRow + bCol * K, K);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        }
    }

    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::store_matrix_sync(C + cRow * N + cCol, c_frag, N, wmma::mem_row_major);
    }
}

#endif // __CUDA_ARCH__ >= 800

#endif // HOPPER_AMPERE_SOL_CUH
