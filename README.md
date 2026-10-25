# CUDA GEMM Optimization 🚀

Repositório voltado ao estudo, benchmark e otimização progressiva de kernels de multiplicação de matrizes (**GEMM**: $C = \alpha \cdot (A \times B) + \beta \cdot C$) em CUDA, evoluindo de uma implementação escalar simples até aceleração por hardware nativa com **Tensor Cores (WMMA)** e suporte a matrizes de dimensões arbitrárias.

![GEMM Performance](docs/gemm_performance.png)

---

## 📊 Benchmark & Resultados (Tesla T4 / L4)

Resultados medidos para matrizes de alta dimensão ($4096 \times 4096$) e dimensões assimétricas genéricas:

| Kernel | Arquitetura / Técnica | Precisão | Tempo (ms) | Desempenho (GFLOPS) | Speedup vs Naive | Status / Validação |
|---|---|---|---|---|---|---|
| **Kernel 00 (Naive)** | Global Memory Scalar | FP32 | 222.12 ms | 618.75 GFLOPS | $1.0\times$ | PASS ($0.00$) |
| **Kernel 01 (Shared Mem)** | Shared Memory Tiling 2D | FP32 | 147.55 ms | 931.48 GFLOPS | $1.51\times$ | PASS ($0.00$) |
| **Kernel 02 (2D Reg + float4)** | Register Tiling + Vectorization + Pad | FP32 | **33.06 ms** | **4156.97 GFLOPS** | **$6.71\times$** | **PASS ($0.00$)** |
| **Kernel 03 (Double Buffering)** | Software Pipelining | FP32 | 33.93 ms | 4050.41 GFLOPS | $6.55\times$ | PASS ($0.00$) |
| **cuBLAS SGEMM (Ref)** | NVIDIA cuBLAS | FP32 | 29.75 ms | 4619.35 GFLOPS | $7.46\times$ | *Reference* |
| **Kernel 04 (FP16 SIMD)** | `half2` (`__hfma2`) | FP16 | 16.80 ms | 8120.50 GFLOPS | $13.1\times$ | PASS |
| **Kernel 05 (WMMA Tensor Cores)** | Hardware Tensor Cores (`mma.sync`) | FP16 | **7.19 ms** | **19120.63 GFLOPS** | **$30.9\times$** | **PASS** |
| **cuBLAS HGEMM (Ref)** | NVIDIA cuBLAS Tensor Cores | FP16 | 3.91 ms | 35112.04 GFLOPS | $56.7\times$ | *Reference* |
| **Kernel 06 (Generic GEMM)** | Boundary Checks + Coeficientes $\alpha/\beta$ | FP32 | **2.43 ms** | **2465.40 GFLOPS** | **$1.72\times$ vs cuBLAS** | **PASS ($0.0137$)** |

---

## 🛠️ Tecnologias e Arquiteturas Utilizadas

1. **Global Memory Access Coalescing:** Leitura contígua de memória em rajadas (*bursts* de 128 bytes).
2. **Shared Memory Tiling (2D):** Armazenamento temporário em L1/Shared Memory para reutilização de blocos.
3. **Register Tiling (2D):** Aumento da densidade de computação por thread em registradores de alta velocidade.
4. **Vetorização de Memória (`float4` / `half2`):** Instruções de 128-bit para maximizar a largura de banda.
5. **Shared Memory Padding:** Eliminação de Bank Conflicts nos acessos simultâneos das threads do Warp.
6. **Software Pipelining / Double Buffering:** Sobreposição de buscas de dados globais com operações de ALU.
7. **Instruções SIMD FP16 (`__hfma2`):** Duas operações de ponto flutuante de 16 bits por ciclo por CUDA Core.
8. **Tensor Cores (`nvcuda::wmma`):** Multiplicação e acumulação matricial nativa por Warp ($16 \times 16 \times 16$).
9. **Boundary Constraints & Scale Factors:** Suporte a matrizes não-quadradas e cálculo de $C = \alpha(A \times B) + \beta C$.

---

## 🛠️ Automação e CI/CD

- **CMake:** Compilação cross-platform declarativa e modularizada.
- **GitHub Actions:** Pipeline automatizada em `.github/workflows/cmake.yml` que compila o projeto a cada push.
- **Scripts em Python:** Geração automatizada de benchmarks e gráficos em `docs/gemm_performance.png`.

---

## 🔨 Como Compilar e Executar

```bash
# 1. Clonar o repositório
git clone [https://github.com/socialmindset20-art/cuda-gemm-optimization.git](https://github.com/socialmindset20-art/cuda-gemm-optimization.git)
cd cuda-gemm-optimization

# 2. Compilar via CMake
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# 3. Executar o benchmark (M N K)
./gemm_bench 4096 4096 4096

# 4. Gerar os gráficos de desempenho
python3 ../scripts/benchmark.py
