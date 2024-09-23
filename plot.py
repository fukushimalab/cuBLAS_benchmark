# plot.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CSVファイルの読み込み
df = pd.read_csv('results.csv')

# データ型ごとに色を分ける
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 各データ型ごとにプロット
for dtype in df['DataType'].unique():
    subset = df[df['DataType'] == dtype]
    plt.plot(subset['MatrixSize'], subset['Median_TIME'], marker='o', label=dtype)

plt.title('MatMul Performance')
plt.xlabel('Matrix Size (N x N)')
plt.ylabel('nano second')
plt.legend()
plt.xscale('log', base=2)
plt.xticks(subset['MatrixSize'], subset['MatrixSize'])
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.savefig('output.png')
plt.show()
