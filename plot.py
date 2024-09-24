import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('results.csv')

sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

for dtype in df['DataType'].unique():
    subset = df[df['DataType'] == dtype]
    plt.plot(subset['MatrixSize'], subset['Median_TIME'], marker='o', label=dtype)

plt.title('MatMul Performance (Linear y-axis)')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('nano second')
plt.legend()
plt.xscale('log', base=2)
plt.xticks(subset['MatrixSize'], subset['MatrixSize'])
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('output_linear.png') 
plt.show()

plt.figure(figsize=(10, 6))

for dtype in df['DataType'].unique():
    subset = df[df['DataType'] == dtype]
    plt.plot(subset['MatrixSize'], subset['Median_TIME'], marker='o', label=dtype)

plt.title('MatMul Performance (Logarithmic y-axis)')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('nano second')
plt.yscale('log')
plt.legend()
plt.xscale('log', base=2)
plt.xticks(subset['MatrixSize'], subset['MatrixSize'])
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('output_log.png')
plt.show()
