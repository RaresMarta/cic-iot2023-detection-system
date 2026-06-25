"""Random-vs-temporal split comparison (MLP only).

Question: does the (capture/file-order) *temporal* split — dropped from the
defaults in commit 6b894a3 — reveal materially different MLP performance than the
headline *random* (stratified row-level) split?

Reuses the exact library code path (SPLIT_FUNCS, fit_preprocess, train_mlp,
metrics5, core.models.evaluate) so the only thing varying between arms is the
split function. Seeds are fixed (SEED) before each MLP fit so the two splits face
identical initialization/shuffling RNG — the only difference is the data split.

Does NOT touch models/ serving artifacts or calibration.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from ids.core.config import SEED, X_COLUMNS_SELECTED, N_EPOCHS
from ids.core.labels import remap_labels
from ids.core.models import evaluate, device
from ids.data.preprocessing import SPLIT_FUNCS, fit_preprocess
from ids.training.data import load_dataset
from ids.training.evaluation import metrics5
from ids.training.trainers import balanced_class_weights, train_mlp, _loader

SPLITS = ['random', 'temporal']
MODES = ['2']
METRIC_KEYS = ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall']


def describe_split(name, tr, va, te, y34):
    def nclasses(idx):
        return len(np.unique(y34[idx]))
    print(f'  [{name}] train={len(tr):,} ({nclasses(tr)} cls) '
          f'val={len(va):,} ({nclasses(va)} cls) '
          f'test={len(te):,} ({nclasses(te)} cls)')


def main():
    t0 = time.time()
    print(f'device={device} | features={len(X_COLUMNS_SELECTED)} | epochs={N_EPOCHS}')
    X_all, y_all_34, source_csv = load_dataset()
    print(f'rows={len(y_all_34):,}\n')

    results: dict = {}

    for split in SPLITS:
        print(f'================ SPLIT: {split} ================', flush=True)
        tr, va, te = SPLIT_FUNCS[split](y_all_34, source_csv, SEED)
        describe_split(split, tr, va, te, y_all_34)
        X_train, X_val, X_test, _prep = fit_preprocess(X_all, tr, va, te)

        for mode in MODES:
            le = LabelEncoder().fit(remap_labels(y_all_34, mode))
            class_names = list(le.classes_)
            K = len(class_names)
            y_tr = np.asarray(le.transform(remap_labels(y_all_34[tr], mode)))
            y_va = np.asarray(le.transform(remap_labels(y_all_34[va], mode)))
            y_te = np.asarray(le.transform(remap_labels(y_all_34[te], mode)))
            cw = balanced_class_weights(y_tr, K)

            # identical RNG state for both split arms -> isolates the split effect
            torch.manual_seed(SEED)
            np.random.seed(SEED)

            print(f'  [MLP {mode}-class] training (K={K})...', flush=True)
            ts = time.time()
            ckpt = f'_tmp_mlp_{split}_{mode}.pt'
            model, history = train_mlp(X_train, y_tr, X_val, y_va, K, cw, ckpt, mode)
            fit_s = time.time() - ts

            for part, Xs, ys in (('train', X_train, y_tr),
                                 ('val', X_val, y_va),
                                 ('test', X_test, y_te)):
                ev = evaluate(model, _loader(Xs, ys, shuffle=False), class_names, device)
                results[(split, mode, part)] = metrics5(
                    ev['y_true'], ev['y_pred'], labels=list(range(K)))
            tt = results[(split, mode, 'test')]
            print(f'    fit={fit_s:.1f}s epochs_ran={len(history.get("train_loss", []))} '
                  f'TEST acc={tt["accuracy"]:.4f} macroF1={tt["macro_f1"]:.4f} '
                  f'wF1={tt["weighted_f1"]:.4f}', flush=True)
        print()

    # ---- Comparison tables (test set) ----
    print('\n' + '=' * 78)
    print('TEST-SET COMPARISON  (random vs temporal) — MLP')
    print('=' * 78)
    for mode in MODES:
        print(f'\n--- {mode}-class ---')
        print(f'{"metric":<18}' + ''.join(f'{s:>12}' for s in SPLITS) + f'{"Δ(temp-rand)":>16}')
        for k in METRIC_KEYS:
            r = results[('random', mode, 'test')][k]
            t = results[('temporal', mode, 'test')][k]
            print(f'{k:<18}{r:>12.4f}{t:>12.4f}{t - r:>+16.4f}')

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

    out = {f'{s}|{m}|{p}': results[(s, m, p)] for (s, m, p) in results}
    with open('compare_splits_mlp_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nwrote compare_splits_mlp_results.json  ({(time.time() - t0) / 60:.1f} min total)')


if __name__ == '__main__':
    main()
