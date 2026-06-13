"""Pure figure functions: data in, matplotlib Figure out.

The notebook is the single place where figures are displayed and saved; these
functions never call ``plt.show()`` or ``savefig`` themselves.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns

from ids.training.calibration import reliability_curve, softmax

MODEL_LABELS = {'mlp': 'MLP', 'rf': 'Random Forest', 'xgb': 'XGBoost'}
MODEL_COLORS = {'mlp': '#5f8dd3', 'rf': '#3aa17e', 'xgb': '#e07a5f'}
PARTITIONS = ('train', 'val', 'test')


def plot_confusion(cm, class_names, title: str | None = None) -> Figure:
    """Row-normalised confusion matrix (diagonal = per-class recall)."""
    cm = np.asarray(cm, dtype=np.float64)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    d = max(5, len(class_names) * 0.7)
    fig = plt.figure(figsize=(d, d * 0.85))
    annot = len(class_names) <= 12

    sns.heatmap(cmn, annot=annot, fmt='.2f' if annot else '', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, vmin=0, vmax=1)
    plt.title(title or 'Confusion matrix (row-normalised = recall)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    return fig


def plot_training_curves(histories: dict, split: str) -> Figure:
    """Train/val loss per epoch, one panel per mode. ``histories[mode] = history``."""
    modes = [m for m, h in histories.items() if h]
    fig, axes = plt.subplots(1, max(len(modes), 1), figsize=(6 * max(len(modes), 1), 4.2),
                             squeeze=False)
    for ax, mode in zip(axes[0], modes):
        h = histories[mode]
        ep = range(1, len(h['train_loss']) + 1)
        ax.plot(ep, h['train_loss'], '-o', ms=3, label='train', color='#e07a5f')
        ax.plot(ep, h['val_loss'], '-o', ms=3, label='val', color='#3aa17e')
        ax.set_title(f'{split} {mode}-class — class-weighted CE')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()

    return fig


def plot_splits_bar(R: dict, mode: str, key: str, title: str, ylabel: str,
                    models=('mlp', 'rf', 'xgb')) -> Figure:
    """Grouped bars of one metric across train/val/test for all models.

    ``R`` is one split's results dict: ``R[(f'mode{mode}', model, partition)][key]``.
    """
    x = np.arange(len(PARTITIONS))
    w = 0.25
    fig = plt.figure(figsize=(8, 5))

    for i, m in enumerate(models):
        vals = [R[(f'mode{mode}', m, s)][key] for s in PARTITIONS]
        plt.bar(x + (i - 1) * w, vals, w, label=MODEL_LABELS[m],
                color=MODEL_COLORS[m], edgecolor='black', linewidth=0.4)

    plt.xticks(x, [s.capitalize() for s in PARTITIONS])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


def plot_permutation_importance(perm: dict, top_n: int = 20) -> Figure:
    """Top-N features by averaged permutation importance (``perm`` from RESULTS)."""
    feats = np.array(perm['features'])
    imp = np.array(perm['importance'])
    order = np.argsort(imp)[::-1][:top_n]
    fig = plt.figure(figsize=(8, 6))

    plt.barh(list(feats[order])[::-1], list(imp[order])[::-1],
             color='#5f8dd3', edgecolor='black', linewidth=0.4)
    plt.xlabel('Permutation importance (mean macro-F1 drop, RF+XGB avg)')
    plt.title(f'Top {top_n} features by permutation importance')
    plt.tight_layout()

    return fig


def plot_reliability(test_logits, test_y, T: float,
                     ece_before: float, ece_after: float,
                     title: str = 'Reliability diagram') -> Figure:
    """Reliability diagram before/after temperature scaling."""
    fig = plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], '--', color='gray', label='perfect calibration')
    xb, yb = reliability_curve(softmax(np.asarray(test_logits)), test_y)
    xa, ya = reliability_curve(softmax(np.asarray(test_logits) / T), test_y)

    plt.plot(xb, yb, '-o', ms=4, color='#e07a5f',
             label=f'uncalibrated (ECE={ece_before:.3f})')
    plt.plot(xa, ya, '-o', ms=4, color='#3aa17e',
             label=f'T-scaled, T={T:.2f} (ECE={ece_after:.3f})')
    plt.xlabel('Confidence (max softmax)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()

    return fig


def plot_family_importance_heatmap(families, matrix, feat_cols,
                                   top_n: int = 20) -> Figure:
    """Per-family permutation importance heatmap (analysis figure)."""
    matrix = np.asarray(matrix)
    top_idx = np.argsort(matrix.mean(axis=0))[-top_n:][::-1]
    fig, ax = plt.subplots(figsize=(14, 6))

    sns.heatmap(matrix[:, top_idx], cmap='viridis',
                xticklabels=[feat_cols[i] for i in top_idx],
                yticklabels=families, ax=ax)
    ax.set_title(f'Per-family permutation importance (top {top_n} features)')
    plt.xticks(rotation=90)
    plt.tight_layout()

    return fig
