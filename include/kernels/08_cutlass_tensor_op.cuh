#ifndef CUTLASS_TENSOR_OP_CUH
#define CUTLASS_TENSOR_OP_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"

/**
 * Kernel GEMM 08: CUTLASS Tensor Cores (OpClassTensorOp)
 * Configuração otimizada para matrizes A (RowMajor) e B (RowMajor) em FP16
 */
inline void run_08_cutlass_tensor_op(
    int M, int N, int K,
    float alpha,
    const __half* A,
    const __half* B,
    float beta,
    float* C) 
{
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = float;
    using ElementAccumulator = float;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using ElementCompute = float;
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm75;

    // Formas geométricas padrão suportadas para Tensor Cores em RowMajor/RowMajor
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 8>; // Instrução nativa compatível

    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute
    >;

    using CutlassGemmTensorOp = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        OperatorClass,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp
    >;

    CutlassGemmTensorOp gemm_operator;

    const cutlass::half_t* ptrA = reinterpret_cast<const cutlass::half_t*>(A);
    const cutlass::half_t* ptrB = reinterpret_cast<const cutlass::half_t*>(B);

    CutlassGemmTensorOp::Arguments arguments{
        {M, N, K},
        {ptrA, K},
        {ptrB, N},
        {C, N},
        {C, N},
        {alpha, beta}
    };

    size_t workspace_size = CutlassGemmTensorOp::get_workspace_size(arguments);
    uint8_t* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(reinterpret_cast<void**>(&workspace_ptr), workspace_size);
    }

    cutlass::Status status = gemm_operator.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS can_implement falhou para as dimensões fornecidas!" << std::endl;
        if (workspace_ptr) cudaFree(workspace_ptr);
        return;
    }

    status = gemm_operator(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Erro de execução no Kernel 08 CUTLASS TensorOp!" << std::endl;
    }

    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }
}

#endif // CUTLASS_TENSOR_OP_CUH
