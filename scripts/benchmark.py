import subprocess
import re
import matplotlib.pyplot as plt
import os

# Dimensões para testar
sizes = [256, 512, 1024, 2048, 3072, 4096]

kernel_00 = []
kernel_01 = []
kernel_02 = []
kernel_03 = []
cublas_ref = []

print("Executando bateria de benchmarks via CMake...")

for N in sizes:
    print(f"Testando N = {N}...")
    result = subprocess.run(["./build/gemm_bench", str(N), str(N), str(N)], capture_output=True, text=True)
    output = result.stdout

    # Extrai os GFLOPS usando Regex
    cublas = re.search(r'\[Reference\] cuBLAS SGEMM:.*\| (\d+\.\d+) GFLOPS', output)
    k0 = re.search(r'\[Kernel 00\] Naive:.*\| (\d+\.\d+) GFLOPS', output)
    k1 = re.search(r'\[Kernel 01\] Shared Mem Tiling:.*\| (\d+\.\d+) GFLOPS', output)
    k2 = re.search(r'\[Kernel 02\] 2D Reg \+ float4 \+ Pad:.*\| (\d+\.\d+) GFLOPS', output)
    k3 = re.search(r'\[Kernel 03\] Double Buffering:.*\| (\d+\.\d+) GFLOPS', output)

    if cublas and k0 and k1 and k2 and k3:
        cublas_ref.append(float(cublas.group(1)) / 1000.0) # Converte para TFLOPS
        kernel_00.append(float(k0.group(1)) / 1000.0)
        kernel_01.append(float(k1.group(1)) / 1000.0)
        kernel_02.append(float(k2.group(1)) / 1000.0)
        kernel_03.append(float(k3.group(1)) / 1000.0)

# Gerar o Gráfico
plt.figure(figsize=(10, 6))
plt.plot(sizes, kernel_00, marker='o', label='Kernel 00: Naive')
plt.plot(sizes, kernel_01, marker='s', label='Kernel 01: Shared Mem Tiling')
plt.plot(sizes, kernel_02, marker='^', linewidth=2.5, label='Kernel 02: 2D Reg + float4 + Pad')
plt.plot(sizes, kernel_03, marker='d', linewidth=2.0, label='Kernel 03: Double Buffering')
plt.plot(sizes, cublas_ref, marker='x', linestyle='--', color='black', label='cuBLAS SGEMM (Reference)')

plt.title('CUDA GEMM Optimization: Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Dimensão da Matriz (N x N)', fontsize=12)
plt.ylabel('Desempenho (TFLOPS)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=10)
plt.tight_layout()

os.makedirs('docs', exist_ok=True)
plt.savefig('docs/gemm_performance.png', dpi=300)
print("Gráfico 'docs/gemm_performance.png' gerado com sucesso!")
