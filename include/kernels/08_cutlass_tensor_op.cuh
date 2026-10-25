#ifndef CUTLASS_TENSOR_OP_CUH
#define CUTLASS_TENSOR_OP_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

// Headers do CUTLASS para Tensor Cores
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"

/**
 * Kernel GEMM 08: CUTLASS Tensor Cores (OpClassTensorOp)
 * Opera em FP16 (entradas) e acumula em FP32 para máxima performance e precisão.
 */
inline void run_08_cutlass_tensor_op(
    int M, int N, int K,
    float alpha,
    const __half* A,
    const __half* B,
    float beta,
    float* C) 
{
    // 1. Definição dos Tipos de Dados
    using ElementInputA = cutlass::half_t;      // FP16 para matriz A
    using ElementInputB = cutlass::half_t;      // FP16 para matriz B
    using ElementOutput = float;               // FP32 para resultado C
    using ElementAccumulator = float;          // FP32 para acumulador intermediário

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::RowMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    // 2. Arquitetura da GPU (Sm75 para Turing/T4, Sm80 para Ampere/A100, Sm89 para L4)
    using ElementCompute = float;
    using OperatorClass = cutlass::arch::OpClassTensorOp; // Habilita Tensor Cores!
    using ArchTag = cutlass::arch::Sm75;                  // Compatibilidade mínima Turing (T4)

    // 3. Geometria das Instantes de Execução (Tiles)
    using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;
    using WarpShape        = cutlass::gemm::GemmShape<64, 64, 32>;
    using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>; // Instrução nativa do hardware

    // 4. Configuração do Epílogo
    using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput,
        128 / cutlass::sizeof_bits<ElementOutput>::value,
        ElementAccumulator,
        ElementCompute
    >;

    // 5. Instanciação do Kernel CUTLASS TensorOp via Template
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

    // Conversão segura de ponteiros para o tipo interno do CUTLASS (half_t)
    const cutlass::half_t* ptrA = reinterpret_cast<const cutlass::half_t*>(A);
    const cutlass::half_t* ptrB = reinterpret_cast<const cutlass::half_t*>(B);

    // 6. Empacotamento dos Argumentos
    CutlassGemmTensorOp::Arguments arguments{
        {M, N, K},
        {ptrA, K},
        {ptrB, N},
        {C, N},
        {C, N},
        {alpha, beta}
    };

    // 7. Alocação de Workspace em Memória GPU
    size_t workspace_size = CutlassGemmTensorOp::get_workspace_size(arguments);
    uint8_t* workspace_ptr = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(reinterpret_cast<void**>(&workspace_ptr), workspace_size);
    }

    cutlass::Status status = gemm_operator.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "O CUTLASS TensorOp não suporta as dimensões/hardware informados!" << std::endl;
        if (workspace_ptr) cudaFree(workspace_ptr);
        return;
    }

    // 8. Lançamento do Kernel na GPU
    status = gemm_operator(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Erro durante a execução do Kernel 08 CUTLASS TensorOp!" << std::endl;
    }

    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }
}

#endif // CUTLASS_TENSOR_OP_CUH
