"""Random-vs-temporal split comparison (Random Forest only).

Question: does the (capture/file-order) *temporal* split — dropped from the
defaults in commit 6b894a3 — reveal materially different performance than the
headline *random* (stratified row-level) split?

Why RF only: it is deterministic (fixed seed, no GPU nondeterminism), CPU-native,
and trains both modes quickly, giving a clean, reproducible leakage signal that
isolates the split effect from model-training noise.

Reuses the exact library code path (SPLIT_FUNCS, fit_preprocess, train_rf,
metrics5) so the only thing that varies between arms is the split function.
Does NOT touch models/ serving artifacts or calibration.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.preprocessing import LabelEncoder

from ids.core.config import SEED, X_COLUMNS_SELECTED
from ids.core.labels import remap_labels
from ids.data.preprocessing import SPLIT_FUNCS, fit_preprocess
from ids.training.data import load_dataset
from ids.training.evaluation import metrics5
from ids.training.trainers import train_rf

SPLITS = ['random', 'temporal']
MODES = ['2', '8']
METRIC_KEYS = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']


def describe_split(name, tr, va, te, y34):
    """Report sizes and how many labels each partition actually contains."""
    def nclasses(idx):
        return len(np.unique(y34[idx]))
    print(f'  [{name}] train={len(tr):,} ({nclasses(tr)} cls) '
          f'val={len(va):,} ({nclasses(va)} cls) '
          f'test={len(te):,} ({nclasses(te)} cls)')


def main():
    t0 = time.time()
    np.random.seed(SEED)
    print(f'features={len(X_COLUMNS_SELECTED)}')
    X_all, y_all_34, source_csv = load_dataset()
    print(f'rows={len(y_all_34):,}\n')

    results: dict = {}

    for split in SPLITS:
        print(f'================ SPLIT: {split} ================')
        tr, va, te = SPLIT_FUNCS[split](y_all_34, source_csv, SEED)
        describe_split(split, tr, va, te, y_all_34)
        X_train, X_val, X_test, _prep = fit_preprocess(X_all, tr, va, te)

        for mode in MODES:
            le = LabelEncoder().fit(remap_labels(y_all_34, mode))
            K = len(le.classes_)
            y_tr = np.asarray(le.transform(remap_labels(y_all_34[tr], mode)))
            y_va = np.asarray(le.transform(remap_labels(y_all_34[va], mode)))
            y_te = np.asarray(le.transform(remap_labels(y_all_34[te], mode)))

            print(f'  [RF {mode}-class] training (K={K})...', flush=True)
            ts = time.time()
            rf = train_rf(X_train, y_tr, mode)
            fit_s = time.time() - ts

            for part, Xs, ys in (('train', X_train, y_tr),
                                 ('val', X_val, y_va),
                                 ('test', X_test, y_te)):
                m = metrics5(ys, rf.predict(Xs), labels=list(range(K)))
                results[(split, mode, part)] = m
            tt = results[(split, mode, 'test')]
            print(f'    fit={fit_s:.1f}s  TEST acc={tt["accuracy"]:.4f} '
                  f'macroF1={tt["macro_f1"]:.4f} wF1={tt["weighted_f1"]:.4f}',
                  flush=True)
        print()

    # ---- Comparison tables (test set) ----
    print('\n' + '=' * 78)
    print('TEST-SET COMPARISON  (random vs temporal)')
    print('=' * 78)
    for mode in MODES:
        print(f'\n--- {mode}-class ---')
        hdr = f'{"metric":<18}' + ''.join(f'{s:>12}' for s in SPLITS) + f'{"Δ(temp-rand)":>16}'
        print(hdr)
        for k in METRIC_KEYS:
            r = results[('random', mode, 'test')][k]
            t = results[('temporal', mode, 'test')][k]
            print(f'{k:<18}' + f'{r:>12.4f}{t:>12.4f}' + f'{t - r:>+16.4f}')

    # ---- Train->test generalization gap (leakage proxy) ----
    print('\n' + '=' * 78)
    print('TRAIN->TEST GAP in macro_f1 (larger gap = more optimistic / leaky)')
    print('=' * 78)
    print(f'{"split/mode":<16}{"train":>10}{"test":>10}{"gap":>10}')
    for split in SPLITS:
        for mode in MODES:
            tr_f1 = results[(split, mode, 'train')]['macro_f1']
            te_f1 = results[(split, mode, 'test')]['macro_f1']
            print(f'{split+"/"+mode:<16}{tr_f1:>10.4f}{te_f1:>10.4f}{tr_f1 - te_f1:>10.4f}')

    # ---- Persist ----
    out = {f'{s}|{m}|{p}': results[(s, m, p)]
           for (s, m, p) in results}
    with open('compare_splits_rf_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote compare_splits_rf_results.json  ({(time.time() - t0) / 60:.1f} min total)')


if __name__ == '__main__':
    main()
