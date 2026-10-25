# CUDA GEMM Optimization 🚀

Repositório dedicado ao estudo e implementação progressiva de kernels de multiplicação de matrizes (**GEMM**: $C = A \times B$) em CUDA, evoluindo do código *Naive* básico até a aceleração por hardware nativa com **Tensor Cores (WMMA)**.

![GEMM Performance](docs/gemm_performance.png)

---

## 📊 Benchmark & Resultados (Tesla T4 / L4 — Matrizes $4096 \times 4096$)

| Kernel | Precisão / Nível | Tempo (ms) | Desempenho (GFLOPS) | Speedup vs Naive | Validação vs Reference |
|---|---|---|---|---|---|
| **Kernel 00 (Naive)** | FP32 (Scalar) | 222.12 ms | 618.75 GFLOPS | $1.0\times$ | PASS ($0.00$) |
| **Kernel 01 (Shared Mem Tiling)** | FP32 (Block Tiling) | 147.55 ms | 931.48 GFLOPS | $1.51\times$ | PASS ($0.00$) |
| **Kernel 02 (2D Reg + float4 + Pad)** | FP32 (Vectorized) | **33.06 ms** | **4156.97 GFLOPS** | **$6.71\times$** | **PASS ($0.00$)** |
| **Kernel 03 (Double Buffering)** | FP32 (Pipelined) | 33.93 ms | 4050.41 GFLOPS | $6.55\times$ | PASS ($0.00$) |
| **cuBLAS SGEMM (Reference)** | FP32 | 29.75 ms | 4619.35 GFLOPS | $7.46\times$ | *Reference* |
| **Kernel 04 (FP16 half2 `__hfma2`)** | FP16 (SIMD) | 16.80 ms | 8120.50 GFLOPS | $13.1\times$ | PASS |
| **Kernel 05 (WMMA Tensor Cores)** | FP16 Hardware | **7.19 ms** | **19120.63 GFLOPS** | **$30.9\times$** | **PASS** |
| **cuBLAS HGEMM (Reference)** | FP16 Hardware | 3.91 ms | 35112.04 GFLOPS | $56.7\times$ | *Reference* |

---

## 🛠️ Otimizações Implementadas

1. **Global Memory Access Coalescing:** Leitura contígua em memória global para maximizar a largura de banda.
2. **Shared Memory Tiling (2D):** Armazenamento em cache L1/Shared Memory para reutilização de dados do bloco.
3. **Register Tiling (2D):** Aumento do volume de trabalho processado por thread em registradores.
4. **Vetorização de Memória (`float4` / `half2`):** Transmissões de $128\text{-bit}$ para evitar estolagem na instrução de carga.
5. **Shared Memory Padding:** Prevenção de Bank Conflicts em acessos à memória compartilhada.
6. **Software Pipelining / Double Buffering:** Overlapping de transferências de memória global/compartilhada com computação em instâncias alternadas.
7. **Instruções SIMD FP16 (`__hfma2`):** $2\times$ operações de 16 bits por ciclo de instrução no CUDA Core.
8. **Tensor Cores (`nvcuda::wmma`):** Aceleração nativa via hardware de multiplicação matricial por Warp ($16 \times 16 \times 16$).

---

## 🔨 Como Compilar e Executar

### Pré-requisitos
- **CUDA Toolkit** ($\ge 11.0$)
- **CMake** ($\ge 3.18$)
- Compilador **C++17** (`g++` / `clang`)
- GPU NVIDIA compatível (Compute Capability $\ge 7.5$)

### Passo a Passo

```bash
# 1. Clonar o repositório
git clone [https://github.com/socialmindset20-art/cuda-gemm-optimization.git](https://github.com/socialmindset20-art/cuda-gemm-optimization.git)
cd cuda-gemm-optimization

# 2. Criar diretório de build e compilar via CMake
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 3. Executar o benchmark individual
./gemm_bench 4096 4096 4096

# 4. Rodar a bateria de testes e gerar o gráfico
python3 ../scripts/benchmark.py
