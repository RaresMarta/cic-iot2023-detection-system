"""Regression guard for the load_dataset double-log1p bug.

The Preprocessor (ids/core/preprocessor.py) is the single source of truth for
preprocessing: it applies inf->null, log1p, median impute, RobustScaler. Serving
(ids/runtime/predictor.py) feeds RAW features straight into Preprocessor.transform.

For training to match serving, load_dataset must ALSO hand RAW features to
fit_preprocess -- it must NOT pre-apply log1p (or inf->null). If it does, the
model trains on log1p(log1p(x)) while serving applies log1p(x): a silent
train/serve mismatch (the same class of bug as the stale preprocessor).
"""
import numpy as np, polars as pl
from ids.training.data import load_dataset
from ids.core.config import (
    X_COLUMNS_SELECTED, LOG_COLUMNS_SELECTED, Y_COLUMN, PARQUET_PATH,
)


def test_load_dataset_returns_raw_untransformed_features():
    df = pl.read_parquet(str(PARQUET_PATH)).drop_nulls(subset=[Y_COLUMN])
    raw = df.select(list(X_COLUMNS_SELECTED)).to_numpy().astype(np.float32)
    X_all, _, _ = load_dataset()
    assert X_all.shape == raw.shape
    col = LOG_COLUMNS_SELECTED[0]
    idx = list(X_COLUMNS_SELECTED).index(col)
    # load_dataset must NOT have applied log1p: its output equals the raw column.
    assert np.allclose(X_all[:1000, idx], raw[:1000, idx], equal_nan=True), (
        f"load_dataset pre-transformed '{col}' (double-log1p bug): "
        f"got {X_all[:3, idx]}, raw {raw[:3, idx]}"
    )
