#ifndef CUTLASS_BASIC_CUH
#define CUTLASS_BASIC_CUH

#include <cuda_runtime.h>
#include <iostream>

// Headers da biblioteca CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

/**
 * Kernel GEMM 07: CUTLASS Device-level API (Basic GEMM)
 * Utiliza as abstrações em C++ do CUTLASS para FP32
 */
inline void run_07_cutlass_basic(
    int M, int N, int K,
    float alpha,
    const float* A,
    const float* B,
    float beta,
    float* C) 
{
    // 1. Definição da Estrutura de Tipos e Layouts
    using ElementInputA = float;               // Tipo dos elementos de A (FP32)
    using ElementInputB = float;               // Tipo dos elementos de B (FP32)
    using ElementOutput = float;               // Tipo dos elementos de C (FP32)
    using ElementAccumulator = float;          // Tipo do acumulador da ALU (FP32)

    using LayoutInputA = cutlass::layout::RowMajor;  // Matriz A em Row-Major
    using LayoutInputB = cutlass::layout::RowMajor;  // Matriz B em Row-Major
    using LayoutOutput = cutlass::layout::RowMajor;  // Matriz C em Row-Major

    // 2. Instanciação do Kernel CUTLASS via Template
    using CutlassGemm = cutlass::gemm::device::Gemm<
        ElementInputA,
        LayoutInputA,
        ElementInputB,
        LayoutInputB,
        ElementOutput,
        LayoutOutput,
        ElementAccumulator
    >;

    CutlassGemm gemm_operator;

    // 3. Empacotamento dos Argumentos de Execução
    CutlassGemm::Arguments arguments{
        {M, N, K},                      // Dimensões do problema
        {A, K},                         // TensorRef para A (ponteiro + stride lda)
        {B, N},                         // TensorRef para B (ponteiro + stride ldb)
        {C, N},                         // TensorRef para C (ponteiro de origem + stride ldc)
        {C, N},                         // TensorRef para C (ponteiro de destino + stride ldc)
        {alpha, beta}                   // Epílogo com coeficientes escalares alpha e beta
    };

    // 4. Alocação de Workspace em Memória de GPU via CUDA Runtime
    size_t workspace_size = CutlassGemm::get_workspace_size(arguments);
    uint8_t* workspace_ptr = nullptr;
    
    if (workspace_size > 0) {
        cudaMalloc(reinterpret_cast<void**>(&workspace_ptr), workspace_size);
    }

    cutlass::Status status = gemm_operator.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "O CUTLASS não suporta a configuração informada!" << std::endl;
        if (workspace_ptr) cudaFree(workspace_ptr);
        return;
    }

    // 5. Lançamento do Kernel na GPU
    status = gemm_operator(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Erro durante a execução do Kernel CUTLASS!" << std::endl;
    }

    if (workspace_ptr) {
        cudaFree(workspace_ptr);
    }
}

#endif // CUTLASS_BASIC_CUH
