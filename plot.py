import pandas as pd
import matplotlib
# ← GUI バックエンドを切り、ファイル出力専用に。
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_all():
    # --- データ読み込み＆pivot（一度だけ） ---
    df = pd.read_csv('results.csv')
    # 行：MatrixSize、列：DataType、値：GFLOPS / Median_TIME_us
    gflops = df.pivot(index='MatrixSize', columns='DataType', values='GFLOPS')
    time   = df.pivot(index='MatrixSize', columns='DataType', values='Median_TIME_us')

    # 共通 x 軸値（等間隔カテゴリ）
    sizes = gflops.index.values
    x = range(len(sizes))

    # 描画共通設定
    grid_opts = dict(which='both', linestyle='--', linewidth=0.5)
    save_opts = dict(dpi=80)  # ←DPI下げてファイル書き出しを高速化

    # --- A) GFLOPS ---
    fig, ax = plt.subplots(figsize=(8,5))
    for col in gflops.columns:
        ax.plot(x, gflops[col].values, marker='o', label=col)
    ax.set_xticks(x);          ax.set_xticklabels(sizes)
    ax.set_title('MatMul Performance: GFLOPS')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('GFLOPS')
    ax.grid(**grid_opts)
    fig.tight_layout()
    fig.savefig('png/output_gflops.png', **save_opts)
    fig.savefig('pdf/output_gflops.pdf', **save_opts)
    
    plt.close(fig)

    # --- B) Time（線形 y） ---
    fig, ax = plt.subplots(figsize=(8,5))
    for col in time.columns:
        ax.plot(x, time[col].values, marker='o', label=col)
    ax.set_xticks(x);          ax.set_xticklabels(sizes)
    ax.set_title('Execution Time (Linear)')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Median Time (µs)')
    ax.grid(**grid_opts)
    fig.tight_layout()
    fig.savefig('png/output_time_linear.png', **save_opts) 
    fig.savefig('pdf/output_time_linear.pdf', **save_opts)
    
    plt.close(fig)

    # --- C) Time（対数 y） ---
    fig, ax = plt.subplots(figsize=(8,5))
    for col in time.columns:
        ax.plot(x, time[col].values, marker='o', label=col)
    ax.set_xticks(x);          ax.set_xticklabels(sizes)
    ax.set_yscale('log')
    ax.set_title('Execution Time (Log Scale)')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Median Time (µs)')
    ax.grid(**grid_opts)
    fig.tight_layout()
    fig.savefig('output_time_log.png', **save_opts)
    plt.close(fig)

if __name__ == '__main__':
    plot_all()

