"""Standalone inference-latency benchmark for the demo / defense (NFR-1).

Produces the p50/p95/p99 per-batch latency table and flows/sec throughput for the
served models, end-to-end over (RobustScaler.transform + forward pass) — exactly the
span NFR-1 is defined on (CSV/feature parsing and SHAP excluded, as in the thesis).

Why this exists separately from ids.training.benchmark.load_mlp: load_mlp builds the
net with the default hidden sizes [128,64] and crashes on the tuned [512,192]
checkpoints. Here we load the model the same correct way the serving predictor does
(MLPClassifier, which infers hidden sizes from the checkpoint), so it actually runs
and benchmarks the exact artifact the demo serves.

Run:
    python -m scripts.benchmark_latency
    python -m scripts.benchmark_latency --runs 2000 --modes 2 8
"""
from __future__ import annotations

import argparse
import platform

import numpy as np
import polars as pl
import torch

from ids.core.config import (
    PARQUET_PATH, MODELS_DIR, X_COLUMNS_SELECTED, BATCH_SIZES, N_WARMUP, N_RUNS, SEED,
)
from ids.training.benchmark import benchmark_model
from ids.runtime.predictor import MLPClassifier


def load_pool(n: int = 10_000) -> np.ndarray:
    """Unscaled feature pool (the 25 served features) to sample batches from."""
    df = (
        pl.scan_parquet(PARQUET_PATH)
        .select(X_COLUMNS_SELECTED)
        .head(n)
        .collect()
    )
    pool = df.to_numpy().astype(np.float32)
    return np.where(np.isnan(pool), 0.0, pool)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--modes', nargs='+', default=['2', '8'])
    ap.add_argument('--runs', type=int, default=N_RUNS)
    ap.add_argument('--warmup', type=int, default=N_WARMUP)
    ap.add_argument('--batches', nargs='+', type=int, default=BATCH_SIZES)
    args = ap.parse_args()

    # NFR-1 is specified on a consumer laptop *CPU*; pin to CPU for a comparable number.
    device = torch.device('cpu')
    pool = load_pool()

    print(f'Machine : {platform.processor() or platform.machine()} | '
          f'Python {platform.python_version()} | torch {torch.__version__} | device cpu')
    print(f'Protocol: {args.warmup} warmup + {args.runs} timed passes/batch | '
          f'pool={pool.shape[0]} flows | span = RobustScaler.transform + forward')
    print(f'NFR-1   : p95 < 100 ms end-to-end on CPU at any batch size\n')

    header = f'{"Mode":<6}{"Batch":<8}{"p50 ms":<10}{"p95 ms":<10}{"p99 ms":<10}{"flows/sec":<14}{"NFR-1":<8}'
    print(header)
    print('-' * len(header))

    worst_p95 = 0.0
    for mode in args.modes:
        clf = MLPClassifier(MODELS_DIR, split='random', mode=mode, device='cpu')
        for bs in args.batches:
            stats = benchmark_model(
                clf.model, clf.preprocessor, pool, bs,
                args.warmup, args.runs, device, SEED,
            )
            worst_p95 = max(worst_p95, stats['p95_ms'])
            ok = 'PASS' if stats['p95_ms'] < 100 else 'FAIL'
            print(f'{mode:<6}{bs:<8}'
                  f'{stats["p50_ms"]:<10.3f}{stats["p95_ms"]:<10.3f}{stats["p99_ms"]:<10.3f}'
                  f'{stats["throughput_flows_per_sec"]:<14,.0f}{ok:<8}')

    verdict = 'DELIVERED' if worst_p95 < 100 else 'NOT MET'
    print(f'\nWorst-case p95 across all batches/modes: {worst_p95:.3f} ms  ->  NFR-1 {verdict}')


if __name__ == '__main__':
    main()
