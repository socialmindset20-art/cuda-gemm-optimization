import subprocess
import re
import matplotlib.pyplot as plt
import os

sizes = [256, 512, 1024, 2048, 3072, 4096]

kernel_00 = []
kernel_01 = []
kernel_02 = []
kernel_03 = []
kernel_04 = []
kernel_05 = []
kernel_06 = []
kernel_07 = []
kernel_08 = []
kernel_09 = []
cublas_sgemm_ref = []
cublas_hgemm_ref = []

print("Executando bateria de benchmarks via CMake (FP32, FP16, WMMA, CUTLASS & PTX SOL)...")

for N in sizes:
    print(f"Testando N = {N}...")
    result = subprocess.run(["./build/gemm_bench", str(N), str(N), str(N)], capture_output=True, text=True)
    output = result.stdout

    cublas_s = re.search(r'\[Reference\] cuBLAS SGEMM:.*\| (\d+\.\d+) GFLOPS', output)
    cublas_h = re.search(r'\[Reference\] cuBLAS HGEMM \(FP16\):.*\| (\d+\.\d+) GFLOPS', output)
    k0 = re.search(r'\[Kernel 00\] Naive:.*\| (\d+\.\d+) GFLOPS', output)
    k1 = re.search(r'\[Kernel 01\] Shared Mem Tiling:.*\| (\d+\.\d+) GFLOPS', output)
    k2 = re.search(r'\[Kernel 02\] 2D Reg \+ float4 \+ Pad:.*\| (\d+\.\d+) GFLOPS', output)
    k3 = re.search(r'\[Kernel 03\] Double Buffering:.*\| (\d+\.\d+) GFLOPS', output)
    k4 = re.search(r'\[Kernel 04\] FP16 half2 \(__hfma2\):.*\| (\d+\.\d+) GFLOPS', output)
    k5 = re.search(r'\[Kernel 05\] WMMA Tensor Cores:.*\| (\d+\.\d+) GFLOPS', output)
    k6 = re.search(r'\[Kernel 06\] Generic GEMM:.*\| (\d+\.\d+) GFLOPS', output)
    k7 = re.search(r'\[Kernel 07\] CUTLASS Basic GEMM:.*\| (\d+\.\d+) GFLOPS', output)
    k8 = re.search(r'\[Kernel 08\] CUTLASS TensorOp \(FP16\):.*\| (\d+\.\d+) GFLOPS', output)
    k9 = re.search(r'\[Kernel 09\] Hopper/Ampere SOL PTX \(FP16\):.*\| (\d+\.\d+) GFLOPS', output)

    if cublas_h: cublas_hgemm_ref.append(float(cublas_h.group(1)) / 1000.0)
    if cublas_s: cublas_sgemm_ref.append(float(cublas_s.group(1)) / 1000.0)
    if k0: kernel_00.append(float(k0.group(1)) / 1000.0)
    if k1: kernel_01.append(float(k1.group(1)) / 1000.0)
    if k2: kernel_02.append(float(k2.group(1)) / 1000.0)
    if k3: kernel_03.append(float(k3.group(1)) / 1000.0)
    if k4: kernel_04.append(float(k4.group(1)) / 1000.0)
    if k5: kernel_05.append(float(k5.group(1)) / 1000.0)
    if k6: kernel_06.append(float(k6.group(1)) / 1000.0)
    if k7: kernel_07.append(float(k7.group(1)) / 1000.0)
    if k8: kernel_08.append(float(k8.group(1)) / 1000.0)
    if k9: kernel_09.append(float(k9.group(1)) / 1000.0)

plt.figure(figsize=(12, 7))

# Kernels FP32
if kernel_00: plt.plot(sizes[:len(kernel_00)], kernel_00, marker='o', label='Kernel 00: Naive (FP32)')
if kernel_01: plt.plot(sizes[:len(kernel_01)], kernel_01, marker='s', label='Kernel 01: Shared Mem (FP32)')
if kernel_02: plt.plot(sizes[:len(kernel_02)], kernel_02, marker='^', linewidth=2.0, label='Kernel 02: 2D Reg + float4 (FP32)')
if kernel_03: plt.plot(sizes[:len(kernel_03)], kernel_03, marker='d', linewidth=2.0, label='Kernel 03: Double Buffering (FP32)')
if kernel_07: plt.plot(sizes[:len(kernel_07)], kernel_07, marker='v', linewidth=2.0, color='orange', label='Kernel 07: CUTLASS Basic (FP32)')
if cublas_sgemm_ref: plt.plot(sizes[:len(cublas_sgemm_ref)], cublas_sgemm_ref, marker='x', linestyle='--', color='gray', label='cuBLAS SGEMM (FP32 Ref)')

# Kernels Tensor Cores / PTX SOL / FP16
if kernel_04: plt.plot(sizes[:len(kernel_04)], kernel_04, marker='P', linewidth=2.0, color='purple', label='Kernel 04: FP16 half2 (__hfma2)')
if kernel_05: plt.plot(sizes[:len(kernel_05)], kernel_05, marker='h', linewidth=2.0, color='green', label='Kernel 05: WMMA Tensor Cores (FP16)')
if kernel_08: plt.plot(sizes[:len(kernel_08)], kernel_08, marker='D', linewidth=2.5, color='cyan', label='Kernel 08: CUTLASS TensorOp (FP16)')
if kernel_09: plt.plot(sizes[:len(kernel_09)], kernel_09, marker='s', linewidth=2.5, color='red', label='Kernel 09: Hopper/Ampere SOL PTX (FP16)')
if cublas_hgemm_ref: plt.plot(sizes[:len(cublas_hgemm_ref)], cublas_hgemm_ref, marker='*', linestyle='--', color='magenta', label='cuBLAS HGEMM (FP16 Ref)')

plt.title('CUDA GEMM Optimization: Native CUDA vs CUTLASS vs PTX SOL vs cuBLAS', fontsize=14, fontweight='bold')
plt.xlabel('Dimensão da Matriz (N x N)', fontsize=12)
plt.ylabel('Desempenho (TFLOPS)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=9, loc='upper left')
plt.tight_layout()

os.makedirs('docs', exist_ok=True)
plt.savefig('docs/gemm_performance.png', dpi=300)
print("Gráfico 'docs/gemm_performance.png' gerado com sucesso!")
