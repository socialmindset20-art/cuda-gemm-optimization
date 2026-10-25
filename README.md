# CUDA GEMM Optimization 🚀

Repositório dedicado ao estudo e implementação progressiva de kernels de multiplicação de matrizes (**GEMM**: $C = A \times B$) em CUDA, evoluindo do código *Naive* básico até otimizações de nível de produção em **FP32** e **FP16**.

![GEMM Performance](docs/gemm_performance.png)

---

## 📊 Benchmark & Resultados (Tesla T4 / L4 — Matrizes $4096 \times 4096$)

| Kernel | Precisão | Tempo (ms) | Desempenho (GFLOPS) | Speedup vs Naive | Validação vs Reference |
|---|---|---|---|---|---|
| **Kernel 00 (Naive)** | FP32 | 222.12 ms | 618.75 GFLOPS | $1.0\times$ | PASS ($0.00$) |
| **Kernel 01 (Shared Mem Tiling)** | FP32 | 147.55 ms | 931.48 GFLOPS | $1.51\times$ | PASS ($0.00$) |
| **Kernel 02 (2D Reg + float4 + Pad)** | FP32 | **32.94 ms** | **4172.92 GFLOPS** | **$6.70\times$** | **PASS ($0.00$)** |
| **Kernel 03 (Double Buffering)** | FP32 | 33.48 ms | 4105.14 GFLOPS | $6.60\times$ | PASS ($0.00$) |
| **cuBLAS SGEMM (Reference)** | FP32 | 34.57 ms | 3975.63 GFLOPS | $6.39\times$ | *Reference* |
| **Kernel 04 (FP16 half2 `__hfma2`)** | FP16 | **16.80 ms** | **8120.50 GFLOPS** | **$13.1\times$** | **PASS** |
| **cuBLAS HGEMM (Reference)** | FP16 | 15.20 ms | 8980.10 GFLOPS | $14.5\times$ | *Reference* |

---

## 🛠️ Tecnologias e Otimizações Utilizadas

1. **Global Memory Access Coalescing:** Leitura contígua em memória global para maximizar a largura de banda.
2. **Shared Memory Tiling (2D):** Armazenamento em cache L1/Shared Memory para reutilização de dados do bloco.
3. **Register Tiling (2D):** Aumento do volume de trabalho processado por thread em registradores.
4. **Vetorização de Memória (`float4` / `half2`):** Instruções de $128\text{-bit}$ para leitura global/compartilhada.
5. **Shared Memory Padding:** Prevenção de Bank Conflicts em acessos à memória compartilhada.
6. **Software Pipelining / Double Buffering:** Sobreposição de transferências de memória com computação em instâncias alternadas.
7. **Instruções SIMD FP16 (`__hfma2`):** Fusão de multiplicação e adição efetuando $2\times$ operações de 16 bits por ciclo de instrução.

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

# 3. Executar o benchmark (M N K)
./gemm_bench 4096 4096 4096

# 4. Executar os testes automatizados em Python
python3 ../scripts/benchmark.py
