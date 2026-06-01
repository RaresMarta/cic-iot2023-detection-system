"""Inference layer: load scaler + encoder + MLP, classify CSVs the same way the notebook does."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import X_COLUMNS, FLAG_COLUMNS, N_FEATURES
from models import IDSModel

LOG_COLUMNS = [c for c in X_COLUMNS if c not in set(FLAG_COLUMNS)]


class IDSPredictor:
    def __init__(self, models_dir: Path, split: str = 'temporal', mode: str = '2',
                 device: str | None = None):
        self.models_dir = Path(models_dir)
        self.split = split
        self.mode = mode
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.scaler  = joblib.load(self.models_dir / f'scaler_{split}.joblib')
        self.encoder = joblib.load(self.models_dir / f'label_encoder_{split}_{mode}class.joblib')
        n_classes = len(self.encoder.classes_)

        self.model = IDSModel(N_FEATURES, n_classes).to(self.device)
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

    def preprocess(self, df: pl.DataFrame) -> np.ndarray:
        missing = [c for c in X_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f'Input CSV is missing required CIC features: {missing}')

        df = df.select(X_COLUMNS)
        df = df.with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in X_COLUMNS
        ])
        df = df.with_columns([pl.col(c).log1p().alias(c) for c in LOG_COLUMNS])

        X = df.to_numpy().astype(np.float32)
        X = np.where(np.isnan(X), 0.0, X)
        return self.scaler.transform(X).astype(np.float32)

    def predict(self, df: pl.DataFrame) -> dict:
        X = self.preprocess(df)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device))
            probs  = torch.softmax(logits / self.temperature, dim=1).cpu().numpy()
        preds  = probs.argmax(axis=1)
        labels = self.encoder.inverse_transform(preds)
        return {
            'labels':      labels,
            'confidences': probs.max(axis=1),
            'probabilities': probs,
            'class_names': list(self.encoder.classes_),
        }
