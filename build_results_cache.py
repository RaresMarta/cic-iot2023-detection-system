"""Rebuild _results_full.joblib from the trained models already in models/,
without retraining. Reproduces the random split, evaluates each available model
on train/val/test, and writes the metrics cache the notebook loads when
SKIP_TRAINING=True. RF-8class is skipped if its artifact is absent.
"""
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from ids.core.config import X_COLUMNS_SELECTED, SEED, MODELS_DIR
from ids.core.labels import remap_labels
from ids.data.preprocessing import SPLIT_FUNCS
from ids.training.data import load_dataset
from ids.training.evaluation import metrics5, report_and_confusion
from ids.runtime.predictor import MLPClassifier, RFClassifier
from ids.training import artifacts

SPLIT = 'random'
MODES = ['2', '8']
COLS = list(X_COLUMNS_SELECTED)

print('Loading dataset...')
X_all, y_all_34, source_csv = load_dataset()
print(f'  {X_all.shape[0]:,} rows x {X_all.shape[1]} features')

tr, va, te = SPLIT_FUNCS[SPLIT](y_all_34, source_csv, SEED)
parts = {'train': tr, 'val': va, 'test': te}
print(f'  split: train {len(tr):,} | val {len(va):,} | test {len(te):,}')


def df_for(idx):
    return pl.from_numpy(X_all[idx], schema=COLS)


R = {}
ts = joblib.load(MODELS_DIR / 'temperature_scaling.joblib')   # {mode:{T,ece_before,ece_after}}
R['calibration'] = ts

for mode in MODES:
    print(f'\n=== mode {mode} ===')
    K = None
    # ---- candidate predictors for this granularity ----
    preds = {}
    try:
        preds['mlp'] = MLPClassifier(MODELS_DIR, split=SPLIT, mode=mode)
    except Exception as e:
        print(f'  MLP {mode}: skip ({e})')
    if (MODELS_DIR / f'ids_rf_{SPLIT}_{mode}class.joblib').exists():
        preds['rf'] = RFClassifier(MODELS_DIR, kind='rf', split=SPLIT, mode=mode)
    else:
        print(f'  RF {mode}: artifact missing -> skipped')

    if not preds:
        continue
    le = next(iter(preds.values())).encoder
    class_names = list(le.classes_)
    K = len(class_names)
    R[f'class_names_{mode}'] = class_names

    # encoded ground truth per partition
    y_enc = {p: le.transform(remap_labels(y_all_34[idx], mode)) for p, idx in parts.items()}

    for kind, clf in preds.items():
        for p, idx in parts.items():
            out = clf.predict(df_for(idx))
            y_pred = np.asarray(out['probabilities']).argmax(axis=1)
            R[(f'mode{mode}', kind, p)] = metrics5(y_enc[p], y_pred, labels=list(range(K)))
        # test-set report (+ confusion matrix for MLP)
        out_te = clf.predict(df_for(te))
        y_pred_te = np.asarray(out_te['probabilities']).argmax(axis=1)
        rep, cm = report_and_confusion(y_enc['test'], y_pred_te, class_names)
        R[f'{kind}_report_{mode}'] = rep
        if kind == 'mlp':
            R[f'mlp_cm_{mode}'] = cm
        m = R[(f'mode{mode}', kind, 'test')]
        print(f'  {kind.upper()} test: acc={m["accuracy"]:.4f} wF1={m["weighted_f1"]:.4f} '
              f'macroF1={m["macro_f1"]:.4f}')

cache = {'splits': {SPLIT: R}, 'calibration': ts}
joblib.dump(cache, artifacts.RESULTS_CACHE)
print(f'\nWrote {artifacts.RESULTS_CACHE}')
print('Keys present:', sorted(k for k in R if isinstance(k, str)))
