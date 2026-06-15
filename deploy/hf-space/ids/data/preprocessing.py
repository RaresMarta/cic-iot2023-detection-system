"""Data preprocessing: splitting, scaling, and imputation."""

import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit

from ids.core.preprocessor import Preprocessor


def split_random(y_34: np.ndarray, source_csv: np.ndarray, seed: int):
    """Stratified row-level split (70/15/15 train/val/test)."""
    n = len(y_34)
    idx = np.arange(n)
    tmp_idx, test_idx = train_test_split(
        idx, test_size=0.15, stratify=y_34, random_state=seed,
    )
    train_idx, val_idx = train_test_split(
        tmp_idx, test_size=0.15 / 0.85, stratify=y_34[tmp_idx], random_state=seed,
    )
    return train_idx, val_idx, test_idx


def split_per_csv(y_34: np.ndarray, source_csv: np.ndarray, seed: int):
    """GroupShuffleSplit on source CSV (no within-session leakage; temporal-agnostic)."""
    n = len(y_34)
    idx = np.arange(n)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tmp_pos, test_pos = next(gss1.split(idx, y_34, groups=source_csv))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed)
    train_pos, val_pos = next(gss2.split(
        idx[tmp_pos], y_34[tmp_pos], groups=source_csv[tmp_pos],
    ))
    return idx[tmp_pos][train_pos], idx[tmp_pos][val_pos], idx[test_pos]


def split_temporal(y_34: np.ndarray, source_csv: np.ndarray, seed: int):
    """Per-folder temporal split: earliest 70% train, next 15% val, latest 15% test."""
    train_idx, val_idx, test_idx = [], [], []
    for folder in np.unique(y_34):
        folder_pos = np.where(y_34 == folder)[0]
        folder_csvs = source_csv[folder_pos]
        unique_csvs = sorted(set(folder_csvs.tolist()))
        n_csvs = len(unique_csvs)

        if n_csvs >= 3:
            n_train = max(1, int(round(n_csvs * 0.70)))
            n_val = max(1, int(round(n_csvs * 0.15)))
            n_train = min(n_train, n_csvs - 2)
            n_val = min(n_val, n_csvs - n_train - 1)
            train_csvs = set(unique_csvs[:n_train])
            val_csvs = set(unique_csvs[n_train:n_train + n_val])
            for global_pos, csv in zip(folder_pos, folder_csvs):
                if csv in train_csvs:
                    train_idx.append(global_pos)
                elif csv in val_csvs:
                    val_idx.append(global_pos)
                else:
                    test_idx.append(global_pos)
        else:
            order = np.argsort(folder_csvs, kind='stable')
            ordered = folder_pos[order]
            n = len(ordered)
            n_train = int(n * 0.70)
            n_val = int(n * 0.15)
            train_idx.extend(ordered[:n_train].tolist())
            val_idx.extend(ordered[n_train:n_train + n_val].tolist())
            test_idx.extend(ordered[n_train + n_val:].tolist())

    return (
        np.array(train_idx, dtype=np.int64),
        np.array(val_idx, dtype=np.int64),
        np.array(test_idx, dtype=np.int64),
    )


SPLIT_FUNCS = {
    'random': split_random,
    'per_csv': split_per_csv,
    'temporal': split_temporal,
}


def fit_preprocess(X_all: np.ndarray, train_idx: np.ndarray,
                   val_idx: np.ndarray, test_idx: np.ndarray,
                   x_columns: list[str] | None = None,
                   log_columns: list[str] | None = None):
    """Fit imputation + RobustScaler on train only; apply to all splits.

    Uses the consolidated Preprocessor class for identical train/serving behavior.

    Returns:
        (X_train, X_val, X_test, preprocessor)
    """
    if x_columns is None:
        from ids.core.config import X_COLUMNS_SELECTED, LOG_COLUMNS_SELECTED
        x_columns = X_COLUMNS_SELECTED
        log_columns = LOG_COLUMNS_SELECTED

    X_train = X_all[train_idx].copy().astype(np.float32)
    X_val = X_all[val_idx].copy().astype(np.float32)
    X_test = X_all[test_idx].copy().astype(np.float32)

    prep = Preprocessor(x_columns, log_columns)
    X_train, _ = prep.fit(X_train)
    X_val = prep.transform(X_val)
    X_test = prep.transform(X_test)

    return X_train, X_val, X_test, prep
