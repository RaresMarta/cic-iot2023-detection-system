"""Training orchestration: the single place the load -> split -> train ->
evaluate -> calibrate -> persist sequence exists. Callable from the notebook
(``from ids.training import run_training``) and headless (``python -m training``).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from ids.core.config import SEED, MODELS_DIR, X_COLUMNS_SELECTED
from ids.core.labels import remap_labels
from ids.data.preprocessing import SPLIT_FUNCS, fit_preprocess
from ids.core.models import evaluate, device
from ids.training import artifacts
from ids.training.calibration import run_calibration
from ids.training.data import load_dataset
from ids.training.evaluation import metrics5, report_and_confusion, global_permutation_importance
from ids.training.trainers import (
    balanced_class_weights, train_mlp, mlp_logits, train_rf, _loader,
)
from ids.training.tracking import Tracker

__all__ = ['run_training']


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def run_training(splits=('temporal',), modes=('2', '8'),
                 wandb_enabled: bool = False, parquet_path=None) -> dict:
    """Train MLP + Random Forest for every (split, mode), persist everything.

    Per (split, mode): serving artifacts (scaler, encoder, weights, feature
    columns), run artifacts (history + test predictions), MLP logits, metrics on
    train/val/test for all three models. Per split: global permutation importance
    (8-class) and temperature-scaling calibration. The serving calibration
    artifact is written for the FIRST split (the headline one).

    Returns ``{'splits': {split: RESULTS}, 'calibration': calib,
    'models': {(split, mode, kind): model}, 'encoders': {(split, mode): le}}``.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    t0 = time.time()
    tracker = Tracker(enabled=wandb_enabled,
                      name=f"train-{splits[0]}-{'+'.join(str(m) for m in modes)}class")
    print(f'device={device} | features={len(X_COLUMNS_SELECTED)}')

    X_all, y_all_34, source_csv = (load_dataset(parquet_path) if parquet_path else load_dataset())
    print(f'rows={len(y_all_34):,}')

    results_all: dict = {}
    models_mem: dict = {}
    encoders_mem: dict = {}
    calibration: dict = {}

    for split in splits:
        print(f'\n================ SPLIT: {split} ================')
        tr, va, te = SPLIT_FUNCS[split](y_all_34, source_csv, SEED)
        print(f'split: train={len(tr):,} val={len(va):,} test={len(te):,}')
        X_train, X_val, X_test, scaler, _ = fit_preprocess(X_all, tr, va, te)
        artifacts.save_serving_artifacts(split, scaler, X_COLUMNS_SELECTED)

        RESULTS: dict = {}
        trees_for_importance = None

        for mode in modes:
            print(f'\n========== {split} / {mode}-class ==========')
            le = LabelEncoder().fit(remap_labels(y_all_34, mode))
            class_names = list(le.classes_)
            K = len(class_names)
            y_tr = np.asarray(le.transform(remap_labels(y_all_34[tr], mode)))
            y_va = np.asarray(le.transform(remap_labels(y_all_34[va], mode)))
            y_te = np.asarray(le.transform(remap_labels(y_all_34[te], mode)))
            artifacts.save_encoder(le, split, mode)
            encoders_mem[(split, mode)] = le
            RESULTS[f'class_names_{mode}'] = class_names

            cw = balanced_class_weights(y_tr, K)

            # ---- MLP ----
            print('  [MLP] training...')
            model, history = train_mlp(X_train, y_tr, X_val, y_va, K, cw,
                                       artifacts.weights_path(split, mode))
            models_mem[(split, mode, 'mlp')] = model
            evals = {part: evaluate(model, _loader(Xs, ys, shuffle=False), class_names, device)
                     for part, Xs, ys in (('test', X_test, y_te),
                                          ('val', X_val, y_va),
                                          ('train', X_train, y_tr))}
            for part, ev in evals.items():
                RESULTS[(f'mode{mode}', 'mlp', part)] = metrics5(ev['y_true'], ev['y_pred'])
            RESULTS[f'mlp_report_{mode}'] = evals['test']['report']
            RESULTS[f'mlp_cm_{mode}'] = evals['test']['confusion_matrix']
            artifacts.save_run_artifacts(history, evals['test']['y_true'],
                                         evals['test']['y_pred'], split, mode)
            test_logits = mlp_logits(model, X_test)
            artifacts.save_logits(split, mode,
                                  mlp_logits(model, X_val), y_va,
                                  test_logits, y_te)
            tm = RESULTS[(f'mode{mode}', 'mlp', 'test')]
            print(f'    MLP test: acc={tm["accuracy"]:.4f} wF1={tm["weighted_f1"]:.4f}')
            tracker.log_history(split, mode, 'mlp', history)
            tracker.log_eval(split, mode, 'mlp', class_names,
                             evals['test']['y_true'], evals['test']['y_pred'],
                             _softmax(test_logits))

            # ---- Random Forest ----
            print('  [RF] training...')
            rf = train_rf(X_train, y_tr)
            models_mem[(split, mode, 'rf')] = rf
            artifacts.save_tree_model(rf, split, mode, 'rf')
            for part, Xs, ys in (('test', X_test, y_te), ('val', X_val, y_va),
                                 ('train', X_train, y_tr)):
                RESULTS[(f'mode{mode}', 'rf', part)] = metrics5(ys, rf.predict(Xs))
            rf_te_pred = rf.predict(X_test)
            rep, _ = report_and_confusion(y_te, rf_te_pred, class_names)
            RESULTS[f'rf_report_{mode}'] = rep
            tracker.log_eval(split, mode, 'rf', class_names, y_te, rf_te_pred,
                             rf.predict_proba(X_test))

            if mode == '8':
                trees_for_importance = (rf, X_test, y_te)

        # ---- Permutation importance (8-class) ----
        if trees_for_importance is not None:
            print('  permutation importance (8-class)...')
            rf8, Xte8, yte8 = trees_for_importance
            perm = global_permutation_importance(rf8, Xte8, yte8,
                                                 X_COLUMNS_SELECTED, SEED)
            RESULTS['perm_importance_8'] = perm
            artifacts.save_perm_importance(perm)

        # ---- Calibration (serving artifact written for the first split) ----
        calib = run_calibration(split, modes,
                                write_artifact=(split == splits[0]))
        RESULTS['calibration'] = calib
        if split == splits[0]:
            calibration = calib

        results_all[split] = RESULTS

    artifacts.save_results(results_all, calibration)
    artifacts.save_paper_numbers(results_all, calibration, splits[0], modes)

    tracker.log_comparison(results_all, modes)
    tracker.finish()

    print(f'\nDONE in {(time.time() - t0) / 60:.1f} min — wrote '
          f'{artifacts.RESULTS_CACHE}, {artifacts.RESULTS_SUMMARY}, '
          f'{artifacts.PAPER_NUMBERS}')

    return {'splits': results_all, 'calibration': calibration,
            'models': models_mem, 'encoders': encoders_mem}
