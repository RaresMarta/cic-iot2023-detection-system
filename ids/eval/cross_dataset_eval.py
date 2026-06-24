"""Cross-dataset evaluation harness — score foreign-dataset pcaps with the frozen
CIC-IoT-2023 2-class (benign-vs-attack) model. INFERENCE ONLY (no retraining).

Pipeline
--------
    pcap(s) --[dpkt window extractor]--> 25 CIC features
            --[feature-parity gate]----> assert cols == feature_columns.joblib
            --[frozen preprocessor]-----> RobustScaler space
            --[frozen RF or MLP]--------> attack-class score + binary verdict
            --[label_maps.to_binary]----> normalise foreign ground truth
            --[evaluate]----------------> recall / precision / F1 / F2 / CM / PR

The model and scaler are the ones fit on CIC-IoT-2023; we only do inference, so
any numbers produced here measure *transfer* of that frozen model to another
dataset's traffic. Nothing in this module mutates the extractor, models, or
training code.

CLI
---
    python -m ids.eval.cross_dataset_eval \
        --pcaps <dir-or-file> --labels <labels.csv> --dataset <name> --model rf

``--labels`` CSV schema
-----------------------
A CSV with one row per extracted feature window, in the SAME ORDER the extractor
emits windows, with at minimum a ``label`` column carrying the foreign dataset's
raw ground-truth label (string or int). The harness aligns row i of the labels
CSV to window i of the feature matrix, normalises each raw label via
``label_maps.to_binary(dataset, raw)``, and compares.

    label
    Normal
    DDoS
    DoS
    ...

Optional columns are ignored. If the row counts differ, the harness errors loudly
rather than silently truncating — a mismatch means the label CSV was not produced
from the same windowing as the features, which would invalidate every metric.

NOTE ON LABEL PROVENANCE (a real risk): the CIC-IoT-2023 window extractor groups
packets into per-host-pair windows. A foreign dataset's labels are typically
per-flow or per-packet, so producing a per-window label CSV that lines up 1:1
with this extractor's output requires a dataset-specific harness step (label the
packets, then aggregate by the same windowing). That step is the human's
responsibility; this module only consumes the resulting aligned CSV.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from ids.core.config import MODELS_DIR
from ids.runtime.extractor import extract_features as _extract_one
from ids.runtime.predictor import MLPClassifier, RFClassifier

# Binary vocabulary used everywhere downstream.
BENIGN, ATTACK = 'benign', 'attack'

# Packet-window size. The CIC-IoT-2023 authors used 10 for non-flood classes and
# 100 for flood classes; cross-dataset we don't know the class a-priori, so we
# default to 10 (the dataset default) and expose it on the CLI.
DEFAULT_WINDOW = 10


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def model_feature_columns() -> list[str]:
    """The exact 25 feature columns (names + order) the frozen model expects."""
    return list(joblib.load(MODELS_DIR / 'feature_columns.joblib'))


def extract_features(pcap_path_or_dir: str | Path, window: int = DEFAULT_WINDOW,
                     include_partial: bool = True) -> pl.DataFrame:
    """Turn a pcap file OR a directory of pcaps into the 25-feature matrix.

    Wraps ``ids.runtime.extractor.extract_features`` (the custom dpkt windowing
    extractor). For a directory, every ``*.pcap`` / ``*.pcapng`` / ``*.cap`` file
    is extracted and the per-window rows are concatenated (sorted by filename so
    the row order is deterministic and reproducible for label alignment).

    Returns a polars DataFrame whose columns are exactly ``model_feature_columns()``
    in order (the underlying extractor already ``.select(...)``s them).
    """
    p = Path(pcap_path_or_dir)
    cols = model_feature_columns()

    if p.is_dir():
        files = sorted(
            f for f in p.iterdir()
            if f.suffix.lower() in {'.pcap', '.pcapng', '.cap'}
        )
        if not files:
            raise FileNotFoundError(f'No .pcap/.pcapng/.cap files found in {p}')
        frames = [_extract_one(f, window=window, include_partial=include_partial)
                  for f in files]
        df = pl.concat(frames, how='vertical') if frames else pl.DataFrame(
            schema={c: pl.Float64 for c in cols})
    else:
        if not p.exists():
            raise FileNotFoundError(f'pcap not found: {p}')
        df = _extract_one(p, window=window, include_partial=include_partial)

    return df


def check_feature_parity(df: pl.DataFrame, *, raise_on_mismatch: bool = True) -> dict:
    """Assert the extractor's columns exactly match the model's expected 25.

    This is a primary correctness gate: if the live/extracted feature names or
    order drift from ``feature_columns.joblib``, the scaler and model silently
    consume mis-aligned inputs and every prediction is garbage. We check names,
    order, and count.

    Returns a report dict (always), and raises ``AssertionError`` on mismatch
    when ``raise_on_mismatch`` is True.
    """
    expected = model_feature_columns()
    actual = list(df.columns)

    exact = actual == expected
    missing = [c for c in expected if c not in actual]
    extra = [c for c in actual if c not in expected]
    # order mismatch only meaningful when the sets agree
    order_ok = (set(actual) == set(expected)) and exact

    report = {
        'ok': exact,
        'n_expected': len(expected),
        'n_actual': len(actual),
        'missing': missing,
        'extra': extra,
        'order_ok': order_ok,
        'expected': expected,
        'actual': actual,
    }

    if not exact:
        msg = (
            'FEATURE-PARITY MISMATCH: extractor columns != feature_columns.joblib\n'
            f'  expected ({len(expected)}): {expected}\n'
            f'  actual   ({len(actual)}): {actual}\n'
            f'  missing from extractor : {missing}\n'
            f'  extra in extractor     : {extra}\n'
            f'  order matches          : {order_ok}'
        )
        if raise_on_mismatch:
            raise AssertionError(msg)
        report['message'] = msg

    return report


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
_PREDICTORS = {
    'rf':  lambda: RFClassifier(MODELS_DIR, kind='rf', split='random', mode='2'),
    'mlp': lambda: MLPClassifier(MODELS_DIR, split='random', mode='2'),
}


def _attack_index(class_names: list[str]) -> int:
    """Index of the attack class within the encoder's class order."""
    lowered = [str(c).strip().lower() for c in class_names]
    if ATTACK in lowered:
        return lowered.index(ATTACK)
    # fall back: the non-benign column is the attack column (2-class only)
    if BENIGN in lowered and len(lowered) == 2:
        return 1 - lowered.index(BENIGN)
    raise ValueError(f'Cannot locate attack class in encoder classes: {class_names}')


def predict(features_df: pl.DataFrame, model: str = 'rf') -> tuple[np.ndarray, np.ndarray]:
    """Score a feature matrix with the frozen 2-class model.

    Runs the parity gate, then column-aligns -> ``preprocessor.transform`` ->
    model -> probability, all via the existing ``RFClassifier`` / ``MLPClassifier``
    (which own the frozen preprocessor + the checkpoint-inferred MLP arch).

    Args:
        features_df: DataFrame with the 25 CIC feature columns.
        model: ``'rf'`` or ``'mlp'``.

    Returns:
        ``(labels, scores)`` where
            labels : np.ndarray[str] of ``'benign'`` / ``'attack'`` (binary vocab)
            scores : np.ndarray[float] attack-class probability in [0, 1]
    """
    if model not in _PREDICTORS:
        raise ValueError(f"model must be one of {list(_PREDICTORS)}, got {model!r}")

    # Hard correctness gate before any inference.
    check_feature_parity(features_df, raise_on_mismatch=True)

    clf = _PREDICTORS[model]()
    out = clf.predict(features_df)          # dict: labels, probabilities, class_names
    ai = _attack_index(out['class_names'])
    scores = np.asarray(out['probabilities'], dtype=float)[:, ai]
    # Normalise the model's raw label strings ('Attack'/'Benign') to binary vocab.
    labels = np.array(
        [ATTACK if str(l).strip().lower() == ATTACK else BENIGN for l in out['labels']]
    )
    return labels, scores


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _to_binary_int(arr) -> np.ndarray:
    """Map a sequence of {'benign','attack'} (or 0/1) to int {0=benign, 1=attack}."""
    out = np.empty(len(arr), dtype=int)
    for i, v in enumerate(arr):
        s = str(v).strip().lower()
        out[i] = 1 if s in (ATTACK, '1') else 0
    return out


def evaluate(pred_labels, true_labels, scores) -> dict:
    """Binary detection metrics treating ``attack`` as the positive class.

    Args:
        pred_labels: predicted ``'benign'`` / ``'attack'`` per window.
        true_labels: ground-truth ``'benign'`` / ``'attack'`` per window
                     (use ``label_maps.to_binary`` to produce these from a
                     foreign dataset's raw labels).
        scores:      attack-class probability per window (for PR curve / F2 at
                     swept thresholds).

    Returns dict with: recall, precision, f1, f2, confusion_matrix (2x2,
    rows=true [benign, attack], cols=pred), confusion_labels, support, and
    pr_curve (lists of precision / recall / threshold from sklearn).
    """
    y_pred = _to_binary_int(pred_labels)
    y_true = _to_binary_int(true_labels)
    s = np.asarray(scores, dtype=float)

    if not (len(y_pred) == len(y_true) == len(s)):
        raise ValueError(
            f'length mismatch: pred={len(y_pred)} true={len(y_true)} scores={len(s)}')

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # 0=benign, 1=attack

    metrics = {
        'recall':    float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1':        float(fbeta_score(y_true, y_pred, beta=1, pos_label=1, zero_division=0)),
        'f2':        float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)),
        'confusion_matrix': cm.tolist(),
        'confusion_labels': {'rows': ['benign', 'attack'], 'cols': ['benign', 'attack']},
        'support': {'benign': int((y_true == 0).sum()), 'attack': int((y_true == 1).sum())},
        'n': int(len(y_true)),
    }

    # PR curve over swept thresholds — only meaningful if both classes present.
    if len(np.unique(y_true)) == 2:
        prec, rec, thr = precision_recall_curve(y_true, s, pos_label=1)
        metrics['pr_curve'] = {
            # precision_recall_curve returns len(thr)+1 points for prec/rec; the
            # last point (recall=0) has no threshold. Keep them aligned by padding.
            'precision': prec.tolist(),
            'recall': rec.tolist(),
            'thresholds': thr.tolist(),
        }
    else:
        metrics['pr_curve'] = None
        metrics['pr_curve_note'] = (
            'PR curve skipped: ground truth has a single class '
            f'({"attack only" if y_true.all() else "benign only"}).')

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _load_true_labels(labels_csv: Path, dataset: str, n_windows: int) -> np.ndarray:
    from ids.eval.label_maps import to_binary
    ldf = pl.read_csv(labels_csv)
    if 'label' not in ldf.columns:
        raise ValueError(
            f"--labels CSV must have a 'label' column; found {ldf.columns}")
    raw = ldf['label'].to_list()
    if len(raw) != n_windows:
        raise ValueError(
            f'Row-count mismatch: {n_windows} feature windows vs {len(raw)} label '
            f'rows in {labels_csv}. The label CSV must be produced from the SAME '
            f'windowing as the features (one label per window, same order).')
    return np.array([to_binary(dataset, r) for r in raw])


def _print_report(metrics: dict) -> None:
    print('\n=== Cross-dataset binary detection (attack = positive) ===')
    print(f"  windows         : {metrics['n']}  "
          f"(benign={metrics['support']['benign']}, attack={metrics['support']['attack']})")
    print(f"  recall          : {metrics['recall']:.4f}")
    print(f"  precision       : {metrics['precision']:.4f}")
    print(f"  F1              : {metrics['f1']:.4f}")
    print(f"  F2              : {metrics['f2']:.4f}")
    cm = metrics['confusion_matrix']
    print('  confusion (rows=true benign/attack, cols=pred benign/attack):')
    print(f'      benign: {cm[0]}')
    print(f'      attack: {cm[1]}')
    if metrics['pr_curve'] is None:
        print(f"  PR curve        : {metrics.get('pr_curve_note', 'n/a')}")
    else:
        print(f"  PR curve points : {len(metrics['pr_curve']['precision'])}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description='Cross-dataset eval of the frozen CIC-IoT-2023 2-class model.')
    ap.add_argument('--pcaps', required=True,
                    help='pcap file or directory of pcaps to score.')
    ap.add_argument('--labels', required=True,
                    help="CSV with a 'label' column, one row per window (same order as extraction).")
    ap.add_argument('--dataset', required=True,
                    help="dataset name for label normalisation (e.g. bot-iot, ton-iot, iot-23).")
    ap.add_argument('--model', default='rf', choices=['rf', 'mlp'])
    ap.add_argument('--window', type=int, default=DEFAULT_WINDOW,
                    help=f'packets per window (default {DEFAULT_WINDOW}).')
    args = ap.parse_args(argv)

    print(f'[1/4] extracting features from {args.pcaps} (window={args.window}) ...')
    feats = extract_features(args.pcaps, window=args.window)
    print(f'      -> {feats.height} windows x {feats.width} features')

    print('[2/4] feature-parity check ...')
    parity = check_feature_parity(feats, raise_on_mismatch=True)
    print(f"      OK: columns match feature_columns.joblib ({parity['n_actual']}/25)")

    print(f'[3/4] predicting with {args.model} ...')
    pred_labels, scores = predict(feats, model=args.model)

    print(f'[4/4] scoring against {args.labels} ...')
    true_labels = _load_true_labels(Path(args.labels), args.dataset, feats.height)
    metrics = evaluate(pred_labels, true_labels, scores)
    _print_report(metrics)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
