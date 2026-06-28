"""Plot the inference-latency benchmark as a simple line chart — latency vs batch size.

One line per model (2-class, 8-class), four points each (batch 1/32/256/1024), y = p95
latency in ms with a faint p50->p99 band so the tail spread is visible without clutter.
No target line, no pass/fail: just the measured behaviour.

Numbers are taken live from the benchmark so the figure can never drift from the table.

Run:
    python -m scripts.plot_latency
    python -m scripts.plot_latency --runs 2000 --out docs/report/figures/latency.png
"""
from __future__ import annotations

import argparse
import platform

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch

from ids.core.config import (
    PARQUET_PATH, MODELS_DIR, X_COLUMNS_SELECTED, BATCH_SIZES, N_WARMUP, N_RUNS, SEED,
)
from ids.training.benchmark import benchmark_model
from ids.runtime.predictor import MLPClassifier

MODE_LABEL = {'2': '2-class (Benign / Attack)', '8': '8-class (attack families)'}
MODE_COLOR = {'2': '#3b6ea5', '8': '#c05a3e'}  # calm blue / warm terracotta


def load_pool(n: int = 10_000) -> np.ndarray:
    df = pl.scan_parquet(PARQUET_PATH).select(X_COLUMNS_SELECTED).head(n).collect()
    pool = df.to_numpy().astype(np.float32)
    return np.where(np.isnan(pool), 0.0, pool)


def collect(modes, batches, warmup, runs):
    device = torch.device('cpu')
    pool = load_pool()
    out = {}
    for mode in modes:
        clf = MLPClassifier(MODELS_DIR, split='random', mode=mode, device='cpu')
        out[mode] = [benchmark_model(clf.model, clf.preprocessor, pool, bs,
                                     warmup, runs, device, SEED) for bs in batches]
    return out


def plot(data, batches, out: str, warmup: int, runs: int):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(batches))

    # 2-class labels above the marker, 8-class below, so overlapping points stay legible.
    label_dy = {'2': 11, '8': -16}
    for mode, stats in data.items():
        p50 = np.array([s['p50_ms'] for s in stats])
        p95 = np.array([s['p95_ms'] for s in stats])
        p99 = np.array([s['p99_ms'] for s in stats])
        c = MODE_COLOR[mode]
        ax.fill_between(x, p50, p99, color=c, alpha=0.13, zorder=1)  # tail band
        ax.plot(x, p95, '-o', color=c, linewidth=2, markersize=6,
                label=MODE_LABEL[mode], zorder=3)
        for xi, v in zip(x, p95):
            ax.annotate(f'{v:.2f} ms', (xi, v), textcoords='offset points',
                        xytext=(0, label_dy[mode]), ha='center', fontsize=8.5, color=c)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{b:,}' for b in batches])
    ax.set_xlabel('Batch size (flows per inference)')
    ax.set_ylabel('Latency per batch (ms)')
    ax.set_title('Inference latency by batch size', fontsize=12.5, fontweight='bold')
    ax.set_ylim(bottom=0)
    ax.grid(axis='y', alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=9.5, loc='upper left')

    machine = platform.processor() or platform.machine()
    fig.text(0.01, -0.02,
             f'{machine} CPU · line = p95, shaded = p50→p99 · '
             f'scale + forward pass · {warmup} warmup + {runs} timed passes per point',
             fontsize=7.5, color='#666', ha='left')

    plt.savefig(out, dpi=140, bbox_inches='tight')
    print(f'wrote {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=['2', '8'])
    ap.add_argument('--batches', nargs='+', type=int, default=BATCH_SIZES)
    ap.add_argument('--runs', type=int, default=N_RUNS)
    ap.add_argument('--warmup', type=int, default=N_WARMUP)
    ap.add_argument('--out', default='docs/report/figures/latency.png')
    args = ap.parse_args()
    data = collect(args.modes, args.batches, args.warmup, args.runs)
    plot(data, args.batches, args.out, args.warmup, args.runs)


if __name__ == '__main__':
    main()
