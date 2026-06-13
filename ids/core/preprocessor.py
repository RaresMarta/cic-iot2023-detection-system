"""Single source of truth for data preprocessing: used identically by training and serving.

Preprocessing pipeline:
  1. Infinity → null
  2. log1p on continuous features
  3. Median imputation (fit on train, apply to all splits)
  4. RobustScaler (fit on train, apply to all splits)
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import RobustScaler


class Preprocessor:
    """Fit on train data; apply identically in training and serving."""

    def __init__(self, x_columns: list[str], log_columns: list[str]):
        self.x_columns = x_columns
        self.log_columns = log_columns
        self.medians: np.ndarray | None = None
        self.scaler: RobustScaler | None = None

    def fit(self, X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Fit medians and scaler on train split. Returns (X_train_scaled, medians)."""
        X = X_train.copy().astype(np.float32)
        self._apply_log1p(X)
        self._apply_inf_to_null(X)

        self.medians = np.nanmedian(X, axis=0)
        self.medians = np.where(np.isnan(self.medians), 0.0, self.medians)
        self._apply_imputation(X)

        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        return X_scaled, self.medians

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted preprocessing (used in val/test and serving)."""
        if self.medians is None or self.scaler is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        X = X.copy().astype(np.float32)
        self._apply_log1p(X)
        self._apply_inf_to_null(X)
        self._apply_imputation(X)

        return self.scaler.transform(X).astype(np.float32)

    def _apply_inf_to_null(self, X: np.ndarray) -> None:
        """In-place: replace infinities with NaN."""
        X[~np.isfinite(X)] = np.nan

    def _apply_log1p(self, X: np.ndarray) -> None:
        """In-place: apply log1p to log_columns."""
        for col_idx, col_name in enumerate(self.x_columns):
            if col_name in self.log_columns:
                X[:, col_idx] = np.log1p(X[:, col_idx])

    def _apply_imputation(self, X: np.ndarray) -> None:
        """In-place: median imputation using fitted medians."""
        nan_mask = np.isnan(X)
        if nan_mask.any():
            X[nan_mask] = np.take(self.medians, np.where(nan_mask)[1])
