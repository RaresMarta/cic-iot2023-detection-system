"""Retrain MLP + RandomForest (2-class and 8-class) on the benign-augmented parquet.

Mirrors ids.training.run_training's load -> split -> fit_preprocess -> train ->
calibrate -> persist sequence, but writes every artifact to models_aug/ so the
depth-25 baselines in models/ stay frozen for the A/B. Reuses the canonical
trainers/preprocessing verbatim -- the ONLY change vs the baselines is the
training data (more benign rows). max_depth stays 25 (hparams.json, unchanged).

Run: cd <root> && PYTHONPATH=<root> .venv/bin/python scripts/train_augmented.py
"""
import json
import time

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from ids.core.config import SEED, MODELS_DIR, X_COLUMNS_SELECTED
from ids.core.labels import remap_labels
from ids.core.models import evaluate, device
from ids.data.preprocessing import split_random, fit_preprocess
from ids.training import artifacts
from ids.training.calibration import run_calibration
from ids.training.data import load_dataset
from ids.training.evaluation import metrics5
from ids.training.trainers import (
    balanced_class_weights, train_mlp, mlp_logits, train_rf, _loader,
)

AUG_DIR = MODELS_DIR.parent / 'models_aug'
AUG_DIR.mkdir(exist_ok=True)
SPLIT = 'random'
PARQUET = 'data/cic_iot_2023_benignaug.parquet'

torch.manual_seed(SEED)
np.random.seed(SEED)
t0 = time.time()
print(f'device={device} | features={len(X_COLUMNS_SELECTED)} | out={AUG_DIR}')

X_all, y34, src = load_dataset(PARQUET)          # raw features (Preprocessor owns log1p)
print(f'rows={len(y34):,}')
tr, va, te = split_random(y34, src, SEED)
print(f'split: train={len(tr):,} val={len(va):,} test={len(te):,}')
X_tr, X_va, X_te, prep = fit_preprocess(X_all, tr, va, te)
artifacts.save_serving_artifacts(SPLIT, prep, X_COLUMNS_SELECTED, models_dir=AUG_DIR)

summary = {}
for mode in ('2', '8'):
    print(f'\n========== {mode}-class ==========')
    le = LabelEncoder().fit(remap_labels(y34, mode))
    class_names = list(le.classes_)
    K = len(class_names)
    y_tr = np.asarray(le.transform(remap_labels(y34[tr], mode)))
    y_va = np.asarray(le.transform(remap_labels(y34[va], mode)))
    y_te = np.asarray(le.transform(remap_labels(y34[te], mode)))
    artifacts.save_encoder(le, SPLIT, mode, models_dir=AUG_DIR)
    cw = balanced_class_weights(y_tr, K)

    print('  [MLP] training...')
    model, hist = train_mlp(X_tr, y_tr, X_va, y_va, K, cw,
                            artifacts.weights_path(SPLIT, mode, AUG_DIR), mode)
    ev = {p: evaluate(model, _loader(Xs, ys, shuffle=False), class_names, device)
          for p, Xs, ys in (('test', X_te, y_te), ('val', X_va, y_va))}
    m_va = metrics5(ev['val']['y_true'], ev['val']['y_pred'], labels=list(range(K)))
    m_te = metrics5(ev['test']['y_true'], ev['test']['y_pred'], labels=list(range(K)))
    artifacts.save_run_artifacts(hist, ev['test']['y_true'], ev['test']['y_pred'],
                                 SPLIT, mode, models_dir=AUG_DIR)
    artifacts.save_logits(SPLIT, mode, mlp_logits(model, X_va), y_va,
                          mlp_logits(model, X_te), y_te, models_dir=AUG_DIR)
    print(f'    MLP val macroF1={m_va["macro_f1"]:.4f}  test macroF1={m_te["macro_f1"]:.4f}')

    print('  [RF] training...')
    rf = train_rf(X_tr, y_tr, mode)
    artifacts.save_tree_model(rf, SPLIT, mode, 'rf', models_dir=AUG_DIR)
    rf_m_te = metrics5(y_te, rf.predict(X_te), labels=list(range(K)))
    print(f'    RF test macroF1={rf_m_te["macro_f1"]:.4f}')

    summary[mode] = {
        'classes': class_names,
        'mlp_val_macro_f1': m_va['macro_f1'],
        'mlp_test_macro_f1': m_te['macro_f1'],
        'rf_test_macro_f1': rf_m_te['macro_f1'],
    }

print('\n  calibration...')
run_calibration(SPLIT, ('2', '8'), models_dir=AUG_DIR, write_artifact=True)

print(f'\nDONE in {(time.time()-t0)/60:.1f} min')
print('SUMMARY ' + json.dumps(summary))
