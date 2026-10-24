#!/bin/bash
make clean
make
mkdir -p docs/results
./gemm_runner > docs/results/rtx_colab_results.csv
echo "Resultados salvos em docs/results/rtx_colab_results.csv"
