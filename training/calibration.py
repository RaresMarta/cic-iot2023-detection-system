"""Confidence calibration by temperature scaling (Guo et al., 2017).

This is a TRAINING step, not a reporting step: it produces
``models/temperature_scaling.joblib``, which the serving path
(``demo/inference.py``) loads to divide logits before the softmax.
"""
from __future__ import annotations

import joblib
import numpy as np
from scipy.optimize import minimize_scalar

from config import MODELS_DIR
from training import artifacts


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)

    return e / e.sum(axis=1, keepdims=True)


def _nll(T: float, logits: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of the true labels under the temperature-scaled softmax."""
    p = softmax(logits / T)

    return float(-np.mean(np.log(p[np.arange(len(y)), y].clip(1e-12))))


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error: support-weighted |accuracy - confidence| gap."""
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0

    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > 0:
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())

    return float(e)


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray) -> float:
    """Single scalar T > 0 minimising the validation NLL."""
    res = minimize_scalar(_nll, bounds=(0.05, 10.0), method='bounded', args=(val_logits, val_y))

    return float(getattr(res, 'x'))


def reliability_curve(probs: np.ndarray, y: np.ndarray,
                      n_bins: int = 15, min_bin: int = 30):
    """(mean confidence, accuracy) per confidence bin, for reliability diagrams."""
    conf = probs.max(axis=1)
    acc = (probs.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []

    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() > min_bin:
            xs.append(conf[m].mean())
            ys.append(acc[m].mean())

    return xs, ys


def run_calibration(split: str, modes, models_dir=MODELS_DIR,
                    write_artifact: bool = True) -> dict:
    """Fit T per mode from the saved validation logits; measure test ECE before/after.

    Returns ``{mode: {'T', 'ece_before', 'ece_after'}}`` and (by default) writes
    the serving artifact ``temperature_scaling.joblib`` keyed by mode — the exact
    format ``demo/inference.py`` expects.
    """
    calib = {}
    for mode in modes:
        d = np.load(artifacts.logits_path(split, mode, models_dir))
        T = fit_temperature(d['val_logits'], d['val_y'])
        p_before = softmax(d['test_logits'])
        p_after = softmax(d['test_logits'] / T)
        calib[mode] = {
            'T': T,
            'ece_before': ece(p_before, d['test_y']),
            'ece_after': ece(p_after, d['test_y']),
        }
        print(f'  calibration mode {mode}: T*={T:.3f}  '
              f'ECE {calib[mode]["ece_before"]:.4f} -> {calib[mode]["ece_after"]:.4f}')
    if write_artifact:
        joblib.dump(calib, models_dir / 'temperature_scaling.joblib')

    return calib
