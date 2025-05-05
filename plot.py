import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_with_description():
    df = pd.read_csv('results.csv')
    # pivot → NumPy配列アクセスだけに
    gflops = df.pivot(index='MatrixSize', columns='DataType', values='GFLOPS')
    time   = df.pivot(index='MatrixSize', columns='DataType', values='Median_TIME_us')
    sizes = list(gflops.index)
    x = range(len(sizes))

    grid_opts = dict(which='both', linestyle='--', linewidth=0.5)
    save_opts = dict(dpi=300)

    # --- GFLOPS ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in gflops.columns:
        ax.plot(x, gflops[col].values, marker='o', label=col)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_title('MatMul Performance: GFLOPS')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('GFLOPS')
    ax.grid(**grid_opts)
    ax.legend(loc='upper left')   # 凡例は別位置に動かすと見やすいです
    fig.tight_layout()
    fig.savefig('png/output_gflops.png', **save_opts)
    fig.savefig('pdf/output_gflops.pdf', **save_opts)
    plt.close(fig)

    # --- Time (Linear) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in time.columns:
        ax.plot(x, time[col].values, marker='o', label=col)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_title('Execution Time (Linear)')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Median Time (µs)')
    ax.grid(**grid_opts)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('png/output_time_linear.png', **save_opts)
    fig.savefig('pdf/output_time_linear.pdf', **save_opts)
    plt.close(fig)

    # --- Time (Log) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in time.columns:
        ax.plot(x, time[col].values, marker='o', label=col)
    ax.set_xticks(x); ax.set_xticklabels(sizes)
    ax.set_yscale('log')
    ax.set_title('Execution Time (Log Scale)')
    ax.set_xlabel('Matrix Size (N x N)')
    ax.set_ylabel('Median Time (µs)')
    ax.grid(**grid_opts)
    ax.legend(loc='upper left')
    fig.tight_layout()
    fig.savefig('png/output_time_log.png', **save_opts)
    fig.savefig('pdf/output_time_log.pdf', **save_opts)
    plt.close(fig)

if __name__ == '__main__':
    plot_with_description()

