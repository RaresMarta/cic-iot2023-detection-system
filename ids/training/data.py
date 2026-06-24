"""Dataset loading and cleaning for training: cached parquet -> model-ready arrays."""
from __future__ import annotations

import numpy as np
import polars as pl

from ids.core.config import (
    PARQUET_PATH, Y_COLUMN,
    X_COLUMNS_SELECTED,
)


def load_dataset(parquet_path=PARQUET_PATH):
    """Read the cached parquet and return ``(X_all, y_all_34, source_csv)``.

    Selects exactly ``config.X_COLUMNS_SELECTED`` (the 25 trained features)
    regardless of what the parquet contains, so the trained feature set never
    depends on the parquet schema. Features are returned RAW: the
    ``Preprocessor`` (ids/core/preprocessor.py) is the single source of truth
    for inf->null, log1p, median imputation, and scaling, and it is the exact
    same object used at serving time. Applying those transforms here too would
    double-log1p the training data relative to serving (a silent train/serve
    mismatch); leave all preprocessing to ``fit_preprocess`` / ``Preprocessor``.
    """
    df = pl.read_parquet(str(parquet_path))
    missing = [c for c in X_COLUMNS_SELECTED if c not in df.columns]

    if missing:
        raise ValueError(f'Parquet at {parquet_path} is missing {len(missing)} selected feature columns: {missing}. ')

    df = df.drop_nulls(subset=[Y_COLUMN])

    X_all = df.select(X_COLUMNS_SELECTED).to_numpy().astype(np.float32)
    y_all_34 = df[Y_COLUMN].to_numpy()
    source_csv = df['source_csv'].to_numpy()

    return X_all, y_all_34, source_csv
