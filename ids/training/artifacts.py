"""All persistence for the training pipeline lives here.

This module is the contract between training, serving (``demo/``), and the
notebook's figures: every artifact file name and format is defined in exactly
one place. Serving-side names (scaler, encoder, weights, feature columns,
temperature scaling) must not change without updating ``demo/inference.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from ids.core.config import MODELS_DIR

RESULTS_CACHE = Path('_results_full.joblib')
RESULTS_SUMMARY = Path('_results_summary.json')
PAPER_NUMBERS = Path('_paper_numbers.json')


# ── Path helpers (single source of artifact names) ──────────────────────────

def scaler_path(split: str, models_dir=MODELS_DIR) -> Path:
    return models_dir / f'scaler_{split}.joblib'


def encoder_path(split: str, mode: str, models_dir=MODELS_DIR) -> Path:
    return models_dir / f'label_encoder_{split}_{mode}class.joblib'


def weights_path(split: str, mode: str, models_dir=MODELS_DIR) -> Path:
    return models_dir / f'ids_dnn_{split}_{mode}class.pth'


def tree_model_path(split: str, mode: str, kind: str, models_dir=MODELS_DIR) -> Path:
    """Serialised tree baseline (kind 'rf'), servable alongside the MLP."""
    return models_dir / f'ids_{kind}_{split}_{mode}class.joblib'


def run_artifacts_path(split: str, mode: str, models_dir=MODELS_DIR) -> Path:
    return models_dir / f'run_artifacts_{split}_{mode}class.joblib'


def logits_path(split: str, mode: str, models_dir=MODELS_DIR) -> Path:
    return models_dir / f'mlp_logits_{split}_{mode}class.npz'


def feature_columns_path(models_dir=MODELS_DIR) -> Path:
    return models_dir / 'feature_columns.joblib'


def perm_importance_path(models_dir=MODELS_DIR) -> Path:
    return models_dir / 'perm_imp_global_test.joblib'


# ── Save / load ──────────────────────────────────────────────────────────────

def save_serving_artifacts(split: str, scaler, feat_cols, models_dir=MODELS_DIR):
    joblib.dump(scaler, scaler_path(split, models_dir))
    joblib.dump(list(feat_cols), feature_columns_path(models_dir))


def save_tree_model(model, split: str, mode: str, kind: str, models_dir=MODELS_DIR):
    """Persist a trained tree baseline so the analyzer can serve it like the MLP.

    Compressed: a depth-20 RandomForest on ~1M rows is ~170 MB raw; compression
    brings the served artefact down to a deployable size."""
    joblib.dump(model, tree_model_path(split, mode, kind, models_dir), compress=3)


def save_encoder(le, split: str, mode: str, models_dir=MODELS_DIR):
    joblib.dump(le, encoder_path(split, mode, models_dir))


def save_run_artifacts(history, y_true, y_pred, split: str, mode: str,
                       models_dir=MODELS_DIR):
    joblib.dump({'history': history, 'y_true': y_true, 'y_pred': y_pred},
                run_artifacts_path(split, mode, models_dir))


def load_run_artifacts(split: str, mode: str, models_dir=MODELS_DIR) -> dict:
    return joblib.load(run_artifacts_path(split, mode, models_dir))


def save_logits(split: str, mode: str, val_logits, val_y, test_logits, test_y,
                models_dir=MODELS_DIR):
    np.savez(logits_path(split, mode, models_dir),
             val_logits=val_logits, val_y=val_y,
             test_logits=test_logits, test_y=test_y)


def load_logits(split: str, mode: str, models_dir=MODELS_DIR):
    return np.load(logits_path(split, mode, models_dir))


def save_perm_importance(perm: dict, models_dir=MODELS_DIR):
    joblib.dump(perm, perm_importance_path(models_dir))


def load_mlp(split: str, mode: str, n_features: int, models_dir=MODELS_DIR):
    """Reload a trained MLP + its encoder from disk (e.g. for benchmarking)."""
    from ids.core.models import IDSModel, device
    le = joblib.load(encoder_path(split, mode, models_dir))
    model = IDSModel(n_features, len(le.classes_)).to(device)
    model.load_state_dict(torch.load(weights_path(split, mode, models_dir),
                                     map_location=device, weights_only=True))
    model.eval()
    return model, le


# ── Results cache and paper numbers ──────────────────────────────────────────

def save_results(results_all: dict, calibration: dict):
    """Persist the full metrics cache plus a JSON summary (no arrays)."""
    joblib.dump({'splits': results_all, 'calibration': calibration}, RESULTS_CACHE)
    summary = {}
    for split, R in results_all.items():
        for k, v in R.items():
            key = f'{split}|' + ('|'.join(k) if isinstance(k, tuple) else k)
            if isinstance(v, np.ndarray) or str(k).startswith(('mlp_cm', 'perm')):
                continue
            summary[key] = v
    with open(RESULTS_SUMMARY, 'w') as f:
        json.dump(summary, f, indent=2, default=str)


def load_results() -> dict:
    """Load the metrics cache written by the last ``run_training`` run."""
    if not RESULTS_CACHE.exists():
        raise FileNotFoundError(
            f'{RESULTS_CACHE} not found — run `python -m training` (or call '
            f'run_training from the notebook) to produce it.'
        )
    return joblib.load(RESULTS_CACHE)


def save_paper_numbers(results_all: dict, calibration: dict, primary_split: str,
                       modes, models=('mlp', 'rf')):
    """The numbers cited in the thesis text, as JSON, from the primary split."""
    R = results_all[primary_split]
    payload = {
        'split': primary_split,
        'test': {m: {mk: R[(f'mode{m}', mk, 'test')] for mk in models} for m in modes},
        'val': {m: {mk: R[(f'mode{m}', mk, 'val')] for mk in models} for m in modes},
        'calibration': calibration,
    }
    with open(PAPER_NUMBERS, 'w') as f:
        json.dump(payload, f, indent=2, default=str)
