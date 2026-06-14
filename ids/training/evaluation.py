"""Metrics and feature-importance analysis shared by all model families."""
from __future__ import annotations

from typing import cast
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)


def metrics5(y_true, y_pred, labels=None) -> dict:
    """The five aggregate metrics reported for every (model, partition) pair.

    Pass ``labels`` (e.g. ``range(n_classes)``) so a class absent from this
    partition still counts (as 0), keeping macro metrics comparable across
    splits — the grouped (per_csv) split can drop a rare class from a partition.
    """
    return dict(
        accuracy = accuracy_score(y_true, y_pred),
        macro_f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        weighted_f1 = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0),
        macro_precision = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
        macro_recall = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0),
    )


def report_and_confusion(y_true, y_pred, class_names: list) -> tuple[str, np.ndarray]:
    lbls = list(range(len(class_names)))  # all classes, even if absent from this partition
    rep = cast(str, classification_report(y_true, y_pred, labels=lbls, target_names=class_names, zero_division=0, digits=4, output_dict=False))
    cm = confusion_matrix(y_true, y_pred, labels=lbls)

    return rep, cm


def global_permutation_importance(rf, X_test, y_test, feat_cols,
                                  seed: int, sample: int = 10_000,
                                  n_repeats: int = 3) -> dict:
    """Random Forest permutation importance (macro-F1 drop) on a test subsample.

    Falls back to the built-in (Gini) importances if permutation fails,
    flagging the fallback in the returned dict.
    """
    try:
        rng = np.random.default_rng(seed)

        sub = rng.choice(len(X_test), size=min(sample, len(X_test)), replace=False)
        pi_rf = permutation_importance(rf, X_test[sub], y_test[sub], n_repeats=n_repeats, random_state=seed, scoring='f1_macro', n_jobs=1)
        imp = pi_rf.importances_mean  # type: ignore[union-attr]

        return {'features': list(feat_cols), 'importance': imp.tolist()}
    except Exception as e:
        print(f'WARNING: permutation importance failed ({e!r}); '
              f'falling back to built-in importances')
        imp = rf.feature_importances_
        return {'features': list(feat_cols), 'importance': imp.tolist(),
                'fallback': 'builtin'}


def per_family_permutation_importance(rf, X_test, y_test_34, y_test_enc,
                                      family_map: dict, feat_cols, seed: int,
                                      min_samples: int = 100,
                                      n_repeats: int = 5) -> tuple[list, np.ndarray]:
    """Permutation importance computed per attack family (slow; analysis only).

    For each fine-grained attack with at least ``min_samples`` test rows, the
    importance is accumulated into its 8-class family row, then each row is
    normalised. Returns ``(families, matrix)`` with one row per family.
    """
    families = sorted(set(family_map.values()))
    fam_imp = {fam: np.zeros(len(feat_cols)) for fam in families}
    y_test_enc = np.asarray(y_test_enc)
    for attack_name, family in family_map.items():
        mask = y_test_34 == attack_name
        if mask.sum() < min_samples:
            continue

        pi = permutation_importance(rf, X_test[mask], y_test_enc[mask],
                                    n_repeats=n_repeats, random_state=seed, n_jobs=2)
        fam_imp[family] += pi.importances_mean  # type: ignore[union-attr]
    matrix = np.array([fam_imp[fam] / (1 + fam_imp[fam].sum()) for fam in families])

    return families, matrix
