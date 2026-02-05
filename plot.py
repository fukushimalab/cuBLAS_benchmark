import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ensure_out_dirs():
    Path('png').mkdir(exist_ok=True)
    Path('pdf').mkdir(exist_ok=True)

def save_legend(labels, save_opts, legend_cols=1, legend_fontsize=11):
    fig, ax = plt.subplots(figsize=(6, 4))
    handles = []
    for label in labels:
        (line,) = ax.plot([], [], marker='o', label=label)
        handles.append(line)
    ax.legend(
        handles=handles,
        labels=labels,
        fontsize=legend_fontsize,
        loc='center',
        ncol=legend_cols,
        frameon=False,
    )
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('png/output_legend.png', bbox_inches='tight', **save_opts)
    fig.savefig('pdf/output_legend.pdf', bbox_inches='tight', **save_opts)
    plt.close(fig)

def plot_with_description(plot_only=False, legend_cols=1, legend_fontsize=11):
    df = pd.read_csv('results.csv')
    # pivot → NumPy配列アクセスだけに
    gflops = df.pivot(index='MatrixSize', columns='DataType', values='GFLOPS')
    time   = df.pivot(index='MatrixSize', columns='DataType', values='Median_TIME_us')
    sizes = list(gflops.index)
    x = range(len(sizes))

    grid_opts = dict(which='both', linestyle='--', linewidth=0.5)
    save_opts = dict(dpi=300)

    ensure_out_dirs()

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

    if not plot_only:
        save_legend(list(gflops.columns), save_opts, legend_cols, legend_fontsize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--plot-only',
        action='store_true',
        help='plot graphs only (skip legend-only output)',
    )
    parser.add_argument(
        '--legend-cols',
        type=int,
        default=1,
        help='number of columns in legend-only output',
    )
    parser.add_argument(
        '--legend-fontsize',
        type=int,
        default=11,
        help='legend font size (default: 11)',
    )
    args = parser.parse_args()
    plot_with_description(
        plot_only=args.plot_only,
        legend_cols=args.legend_cols,
        legend_fontsize=args.legend_fontsize,
    )
