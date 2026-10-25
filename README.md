# CUDA GEMM Optimization

Repositório dedicado ao estudo, implementação e benchmark incremental de algoritmos de multiplicação de matrizes genérica (**GEMM**: $C = A \times B$) otimizados em CUDA C++, comparados com a biblioteca de referência **cuBLAS**.

## 🚀 Técnicas de Otimização Implementadas

1. **2D Register Tiling ($8 \times 8$):** Reuso de dados diretamente nos registradores de cada thread para elevar a intensidade aritmética.
2. **Shared Memory Tiling ($128 \times 128 \times 16$):** Redução do tráfego com a memória global DRAM.
3. **Vetorização de Memória (`float4` / 128-bit):** Leitura/escrita contínua em blocos de 16 bytes via instruções `LDG.128` e `STG.128`.
4. **Shared Memory Padding:** Inserção de colunas extras no array compartilhado para evitar conflitos de bancos (*bank conflicts*).
5. **Software Pipelining / Double Buffering:** Sobreposição entre computação e transferência de dados da memória global.

---

## 📊 Tabela de Desempenho (GPU Tesla T4/L4 em Matrizes $4096 \times 4096$)

| Kernel | Tempo (ms) | Desempenho (GFLOPS) | Validado vs cuBLAS |
|---|---|---|---|
| **2D Register Tiling (Base)** | 28.15 ms | **4880.7 GFLOPS** | PASS |
| **2D Register Tiling + float4 + Padding** | 34.29 ms | **4007.6 GFLOPS** | PASS |
| **Double Buffering (Pipelining)** | 40.48 ms | **3394.5 GFLOPS** | PASS |
| **cuBLAS Reference** | 29.52 ms | **4654.3 GFLOPS** | Baseline |

---

## 🛠️ Compilação e Execução

```bash
# Compilar o projeto usando nvcc e cuBLAS
nvcc -O3 -Iinclude src/main.cu -o gemm_bench -lcublas

# Executar o benchmark
./gemm_bench
