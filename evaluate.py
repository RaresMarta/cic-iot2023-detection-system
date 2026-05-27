"""Compute train/val/test metrics for MLP, RF, and XGBoost; log charts to wandb."""
from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from preprocessing import fit_preprocess

_MODEL_NAMES  = ("MLP", "RandomForest", "XGBoost")
_SPLIT_TYPES  = ("train", "val", "test")
_COLORS       = {"MLP": "#e07a5f", "RandomForest": "#5f8dd3", "XGBoost": "#3aa17e"}
_METRIC_KEYS  = ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall")


def run_evaluation(
    split_name: str,
    mode: str,
    split_indices: dict,
    X_all: np.ndarray,
    y_all_34: np.ndarray,
    remap_labels: Callable,
    encoders: dict,
    mlp_models: dict,
    mlp_results: dict,
    tree_results: dict,
    tree_models: dict,
    device: torch.device,
) -> dict:
    """Compute train/val/test metrics for all three model families and log to wandb.

    Returns:
        all_metrics[model_name][split_type][metric] → float
    """
    tr_idx, va_idx, te_idx = split_indices[split_name]
    X_train, X_val, X_test, _, _ = fit_preprocess(X_all, tr_idx, va_idx, te_idx)

    le          = encoders[(split_name, mode)]
    y_train_enc = le.transform(remap_labels(y_all_34[tr_idx], mode))
    y_val_enc   = le.transform(remap_labels(y_all_34[va_idx], mode))
    y_test_enc  = le.transform(remap_labels(y_all_34[te_idx], mode))

    def _metrics(y_true, y_pred):
        return {
            "accuracy":        accuracy_score(y_true, y_pred),
            "macro_f1":        f1_score(y_true, y_pred, average="macro",    zero_division=0),
            "weighted_f1":     f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall":    recall_score(y_true, y_pred,    average="macro", zero_division=0),
        }

    all_metrics: dict = {}

    # ── MLP ──────────────────────────────────────────────────────────────────
    print("Computing metrics for MLP...")
    mlp = mlp_models[(split_name, mode)]
    mlp.eval()
    with torch.no_grad():
        tr_pred = mlp(torch.tensor(X_train, dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
        va_pred = mlp(torch.tensor(X_val,   dtype=torch.float32).to(device)).argmax(1).cpu().numpy()
    all_metrics["MLP"] = {
        "train": _metrics(y_train_enc, tr_pred),
        "val":   _metrics(y_val_enc,   va_pred),
        "test":  {k: mlp_results[(split_name, mode)][k] for k in _METRIC_KEYS},
    }

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("Computing metrics for RandomForest...")
    rf = tree_models[(split_name, mode, "rf")]
    all_metrics["RandomForest"] = {
        "train": _metrics(y_train_enc, rf.predict(X_train)),
        "val":   _metrics(y_val_enc,   rf.predict(X_val)),
        "test":  {k: tree_results[(split_name, mode, "rf")][k] for k in _METRIC_KEYS},
    }

    # ── XGBoost ──────────────────────────────────────────────────────────────
    print("Computing metrics for XGBoost...")
    xgb = tree_models[(split_name, mode, "xgb")]
    all_metrics["XGBoost"] = {
        "train": _metrics(y_train_enc, xgb.predict(X_train)),
        "val":   _metrics(y_val_enc,   xgb.predict(X_val)),
        "test":  {k: tree_results[(split_name, mode, "xgb")][k] for k in _METRIC_KEYS},
    }

    del X_train, X_val, X_test

    # Print summary table
    print(f'\n{"Model":<15}{"Train":>10}{"Val":>10}{"Test":>10}')
    print("-" * 45)
    for m in _MODEL_NAMES:
        s = all_metrics[m]
        print(f'{m:<15}{s["train"]["accuracy"]:>10.4f}{s["val"]["accuracy"]:>10.4f}{s["test"]["accuracy"]:>10.4f}')

    _log_to_wandb(split_name, mode, all_metrics)
    return all_metrics


# ── wandb helpers ─────────────────────────────────────────────────────────────

def _log_to_wandb(split_name: str, mode: str, all_metrics: dict) -> None:
    run = wandb.init(
        project="cic-iot2023-ids",
        name=f"{split_name}/{mode}-class/performance-summary",
        job_type="analysis",
    )

    # Summary table
    table = wandb.Table(columns=["Model", "Split", "Accuracy", "MacroF1",
                                  "WeightedF1", "MacroPrecision", "MacroRecall"])
    for m in _MODEL_NAMES:
        for s in _SPLIT_TYPES:
            met = all_metrics[m][s]
            table.add_data(m, s, met["accuracy"], met["macro_f1"],
                           met["weighted_f1"], met["macro_precision"], met["macro_recall"])
    run.log({"performance_summary_table": table})

    # Line plots
    for metric in ("accuracy", "macro_f1", "weighted_f1"):
        fig, ax = plt.subplots(figsize=(10, 6))
        for m in _MODEL_NAMES:
            vals = [all_metrics[m][s][metric] for s in _SPLIT_TYPES]
            ax.plot(_SPLIT_TYPES, vals, marker="o", label=m,
                    linewidth=2.5, markersize=8, color=_COLORS[m])
            for s, v in zip(_SPLIT_TYPES, vals):
                ax.text(list(_SPLIT_TYPES).index(s), v + 0.02, f"{v:.3f}",
                        ha="center", fontsize=9)
        title = metric.replace("_", " ").title()
        ax.set(xlabel="Split", ylabel=title,
               title=f"{title} Across Train/Val/Test Splits", ylim=[0, 1.05])
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        run.log({f"{metric}_across_splits": wandb.Image(fig)})
        plt.close(fig)

    # Grouped bar — model comparison per split
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(_MODEL_NAMES))
    for ax, split_type in zip(axes, _SPLIT_TYPES):
        vals = [all_metrics[m][split_type]["accuracy"] for m in _MODEL_NAMES]
        bars = ax.bar(x, vals, color=[_COLORS[m] for m in _MODEL_NAMES],
                      edgecolor="black", linewidth=0.5)
        ax.set(title=f"{split_type.capitalize()} Accuracy",
               ylabel="Accuracy", ylim=[0, 1.0])
        ax.set_xticks(x); ax.set_xticklabels(_MODEL_NAMES)
        ax.grid(True, alpha=0.2, axis="y")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02, f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    run.log({"model_comparison_by_split": wandb.Image(fig)})
    plt.close(fig)

    # Generalisation analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(_MODEL_NAMES))
    width = 0.25
    for offset, split_type, label, color in zip(
        [-width, 0, width], _SPLIT_TYPES,
        ("Train", "Val", "Test"), ("#3aa17e", "#f2a65a", "#e07a5f"),
    ):
        vals = [all_metrics[m][split_type]["accuracy"] for m in _MODEL_NAMES]
        ax.bar(x_pos + offset, vals, width, label=label,
               color=color, edgecolor="black", linewidth=0.5)
    ax.set(title="Generalization Across Splits: Train vs Val vs Test Accuracy",
           ylabel="Accuracy", ylim=[0, 1.0])
    ax.set_xticks(x_pos); ax.set_xticklabels(_MODEL_NAMES)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()
    run.log({"generalization_analysis": wandb.Image(fig)})
    plt.close(fig)

    wandb.finish()
