import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

df = pd.read_csv('docs/results/rtx_colab_results.csv')

plt.figure()
sns.lineplot(data=df, x='Matrix_Size', y='GFLOPS', hue='Kernel', marker='o', linewidth=2.5)
plt.title('Performance do GEMM: GFLOPS vs Tamanho da Matriz', fontsize=14)
plt.xlabel('Tamanho da Matriz (M=N=K)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.xticks(df['Matrix_Size'].unique())

os.makedirs('docs/results/plots/', exist_ok=True)
plt.savefig('docs/results/plots/gflops_by_matrix_size.png', dpi=300, bbox_inches='tight')
print("Gráfico salvo com sucesso!")
