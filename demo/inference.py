"""Inference layer: load scaler + encoder + MLP, classify CSVs the same way the notebook does."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import polars as pl
import torch
import torch.nn as nn


X_COLUMNS = [
    'Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Variance',
]
N_FEATURES = len(X_COLUMNS)

FLAG_COLUMNS = {
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
    'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP',
    'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP',
    'IPv', 'LLC',
}
LOG_COLUMNS = [c for c in X_COLUMNS if c not in FLAG_COLUMNS]


class IDSModel(nn.Module):
    def __init__(self, n_features: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class IDSPredictor:
    def __init__(self, models_dir: Path, split: str = 'temporal', mode: str = '2',
                 device: str | None = None):
        self.models_dir = Path(models_dir)
        self.split = split
        self.mode = mode
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.scaler = joblib.load(self.models_dir / f'scaler_{split}.joblib')
        self.encoder = joblib.load(self.models_dir / f'label_encoder_{split}_{mode}class.joblib')
        n_classes = len(self.encoder.classes_)

        self.model = IDSModel(N_FEATURES, n_classes).to(self.device)
        state = torch.load(self.models_dir / f'ids_dnn_{split}_{mode}class.pth',
                           map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()

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
        # NaN → 0 (the trained scaler already handles transformed-space values; this is a
        # defensive fill for any residual nulls in the live input)
        X = np.where(np.isnan(X), 0.0, X)
        return self.scaler.transform(X).astype(np.float32)

    def predict(self, df: pl.DataFrame) -> dict:
        X = self.preprocess(df)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        labels = self.encoder.inverse_transform(preds)
        confidences = probs.max(axis=1)
        return {
            'labels': labels,
            'confidences': confidences,
            'probabilities': probs,
            'class_names': list(self.encoder.classes_),
        }
