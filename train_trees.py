"""Train Random Forest and XGBoost baselines and log metrics to wandb."""
from __future__ import annotations

from typing import Callable

import numpy as np
import wandb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight

from preprocessing import fit_preprocess


def evaluate_sklearn(model, X_test: np.ndarray, y_test: np.ndarray, class_names: list) -> dict:
    y_pred = model.predict(X_test)
    return {
        "accuracy":        accuracy_score(y_test, y_pred),
        "macro_f1":        f1_score(y_test, y_pred, average="macro",    zero_division=0),
        "weighted_f1":     f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "macro_precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_recall":    recall_score(y_test, y_pred,    average="macro", zero_division=0),
        "report":          classification_report(y_test, y_pred, target_names=class_names,
                                                 zero_division=0, digits=4),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_true": y_test,
        "y_pred": y_pred,
    }


def train_trees(
    X_all: np.ndarray,
    y_all_34: np.ndarray,
    split_indices: dict,
    splits_to_run: list[str],
    modes_to_run: list[str],
    encoders: dict,
    remap_labels: Callable,
    seed: int = 42,
) -> tuple[dict, dict]:
    """Train RF + XGBoost for every (split, mode) combination.

    Returns:
        tree_results – dict keyed by (split, mode, model_name) → metrics dict
        tree_models  – dict keyed by (split, mode, model_name) → fitted model
    """
    tree_results: dict = {}
    tree_models:  dict = {}

    for split_name in splits_to_run:
        print(f'\n{"=" * 60}\nTREE BASELINES — split: {split_name}\n{"=" * 60}')
        tr_idx, va_idx, te_idx = split_indices[split_name]
        X_train, X_val, X_test, _, _ = fit_preprocess(X_all, tr_idx, va_idx, te_idx)

        for mode in modes_to_run:
            print(f"\n--- {split_name} / {mode}-class ---")
            le = encoders[(split_name, mode)]
            y_train_enc = le.transform(remap_labels(y_all_34[tr_idx], mode))
            y_test_enc  = le.transform(remap_labels(y_all_34[te_idx], mode))
            class_names = list(le.classes_)
            n_classes   = len(class_names)

            # ── Random Forest ────────────────────────────────────────────
            print("  RandomForest...")
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=20,
                class_weight="balanced", n_jobs=-1, random_state=seed,
            )
            rf.fit(X_train, y_train_enc)
            rf_m = evaluate_sklearn(rf, X_test, y_test_enc, class_names)
            print(f"    acc: {rf_m['accuracy']:.4f} | macro-F1: {rf_m['macro_f1']:.4f}")

            _log_tree(split_name, mode, "rf", rf_m,
                      {"model": "RandomForest", "n_estimators": 200,
                       "max_depth": 20, "class_weight": "balanced"})

            # ── XGBoost ──────────────────────────────────────────────────
            print("  XGBoost...")
            present = np.unique(y_train_enc)
            w = compute_class_weight("balanced", classes=present, y=y_train_enc)
            sample_weight = np.ones(len(y_train_enc), dtype=np.float32)
            for cls, wt in zip(present, w):
                sample_weight[y_train_enc == cls] = wt

            xgb_clf = xgb.XGBClassifier(
                n_estimators=300, max_depth=8, learning_rate=0.1,
                tree_method="hist", n_jobs=-1, random_state=seed,
                objective="binary:logistic" if n_classes == 2 else "multi:softprob",
                eval_metric="logloss"        if n_classes == 2 else "mlogloss",
            )
            xgb_clf.fit(X_train, y_train_enc, sample_weight=sample_weight)
            xgb_m = evaluate_sklearn(xgb_clf, X_test, y_test_enc, class_names)
            print(f"    acc: {xgb_m['accuracy']:.4f} | macro-F1: {xgb_m['macro_f1']:.4f}")

            _log_tree(split_name, mode, "xgb", xgb_m,
                      {"model": "XGBoost", "n_estimators": 300,
                       "max_depth": 8, "learning_rate": 0.1, "tree_method": "hist"})

            tree_results[(split_name, mode, "rf")]  = rf_m
            tree_results[(split_name, mode, "xgb")] = xgb_m
            tree_models[(split_name, mode, "rf")]   = rf
            tree_models[(split_name, mode, "xgb")]  = xgb_clf

        del X_train, X_val, X_test

    return tree_results, tree_models


def _log_tree(split_name: str, mode: str, model_key: str, metrics: dict, cfg: dict) -> None:
    run = wandb.init(
        project="cic-iot2023-ids",
        name=f"{split_name}/{mode}-class/{model_key}",
        job_type="baseline",
        config={"split": split_name, "mode": mode, **cfg},
    )
    run.log({
        "test_accuracy":        metrics["accuracy"],
        "test_macro_f1":        metrics["macro_f1"],
        "test_weighted_f1":     metrics["weighted_f1"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall":    metrics["macro_recall"],
    })
    wandb.finish()
