"""Inference layer: load scaler + encoder + model, classify CSVs the same way training did.

Two servable backends share the same preprocessing and the same output contract, so the
web app can swap between them by key without any downstream change:
  IDSPredictor  — the PyTorch MLP, with temperature-scaled (calibrated) confidence.
  TreePredictor — the Random Forest / XGBoost baselines (sklearn ``predict_proba``).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch

from ids.core.config import X_COLUMNS as CONFIG_X_COLUMNS, FLAG_COLUMNS as CONFIG_FLAG_COLUMNS
from ids.core.models import IDSModel


class _BasePredictor:
    """Loads the artefacts every model variant shares (feature columns, scaler, encoder)
    and the preprocessing they must all apply identically. Subclasses load their own model
    and implement ``predict``."""

    def __init__(self, models_dir: Path, split: str = 'temporal', mode: str = '2'):
        self.models_dir = Path(models_dir)
        self.split = split
        self.mode = mode

        # The exact 25-feature column set the models were trained on, persisted at
        # training time; fall back to the full config list only if absent.
        fc_path = self.models_dir / 'feature_columns.joblib'
        self.x_columns = list(joblib.load(fc_path)) if fc_path.exists() else list(CONFIG_X_COLUMNS)
        flags = set(CONFIG_FLAG_COLUMNS)
        self.log_columns = [c for c in self.x_columns if c not in flags]

        self.scaler  = joblib.load(self.models_dir / f'scaler_{split}.joblib')
        self.encoder = joblib.load(self.models_dir / f'label_encoder_{split}_{mode}class.joblib')

    def preprocess(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in self.x_columns if c not in df.columns]
        if missing:
            raise ValueError(f'Input CSV is missing required CIC features: {missing}')

        df = df.select(self.x_columns)
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in self.x_columns
        ])
        df = df.with_columns([pl.col(c).log1p().alias(c) for c in self.log_columns])

        X = df.to_numpy().astype(np.float32)
        X = np.where(np.isnan(X), 0.0, X)

        return self.scaler.transform(X).astype(np.float32)

    def _result(self, probs: np.ndarray) -> dict:
        """Shared output contract: labels, confidences, probabilities, class names."""
        preds = probs.argmax(axis=1)
        return {
            'labels':       self.encoder.inverse_transform(preds),
            'confidences':  probs.max(axis=1),
            'probabilities': probs,
            'class_names':  list(self.encoder.classes_),
        }

    def predict(self, df: pl.DataFrame) -> dict:
        raise NotImplementedError


class IDSPredictor(_BasePredictor):
    """The PyTorch MLP, with optional temperature-scaled (calibrated) confidence."""

    def __init__(self, models_dir: Path, split: str = 'temporal', mode: str = '2',
                 device: str | None = None):
        super().__init__(models_dir, split, mode)
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        n_classes = len(self.encoder.classes_)

        self.model = IDSModel(len(self.x_columns), n_classes).to(self.device)
        state = torch.load(self.models_dir / f'ids_dnn_{split}_{mode}class.pth',
                           map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

        # Optional temperature scaling for calibrated confidence (Guo et al. 2017).
        # Falls back to T=1.0 (no-op) if the calibration artefact is absent.
        self.temperature = 1.0
        temp_path = self.models_dir / 'temperature_scaling.joblib'
        if temp_path.exists():
            try:
                self.temperature = float(joblib.load(temp_path).get(mode, {}).get('T', 1.0))
            except Exception:
                self.temperature = 1.0

    def predict(self, df: pl.DataFrame) -> dict:
        X = self.preprocess(df)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device))
            probs  = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
        return self._result(probs)


class TreePredictor(_BasePredictor):
    """A serialised sklearn RandomForest / XGBoost baseline (kind 'rf' or 'xgb'), served
    with the same preprocessing and output contract as the MLP. No temperature scaling —
    tree ensembles produce their own probability estimates."""

    def __init__(self, models_dir: Path, kind: str, split: str = 'temporal', mode: str = '2'):
        super().__init__(models_dir, split, mode)
        self.kind = kind
        self.model = joblib.load(self.models_dir / f'ids_{kind}_{split}_{mode}class.joblib')

    def predict(self, df: pl.DataFrame) -> dict:
        X = self.preprocess(df)
        raw = np.asarray(self.model.predict_proba(X), dtype=np.float32)

        # predict_proba columns follow the model's own class order; remap to the full
        # encoder class width in case a class was absent from the training fold.
        K = len(self.encoder.classes_)
        if raw.shape[1] == K:
            probs = raw
        else:
            probs = np.zeros((raw.shape[0], K), dtype=np.float32)
            probs[:, np.asarray(self.model.classes_, dtype=int)] = raw

        return self._result(probs)
