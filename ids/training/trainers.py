"""Model training: MLP (PyTorch) and the RF/XGBoost tree baselines.

Hyperparameters are fixed to the values selected by the Optuna search and used
for every reported result; see the thesis methodology chapter.
"""
from __future__ import annotations

import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from ids.core.config import BATCH_SIZE, N_EPOCHS, PATIENCE, LR, SEED
from ids.core.models import IDSDataset, IDSModel, train_model, device


def balanced_class_weights(y_tr: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced-frequency class weights, length ``n_classes`` (absent classes -> 1)."""
    present = np.unique(y_tr)
    w = compute_class_weight('balanced', classes=present, y=y_tr)
    cw = np.ones(n_classes, dtype=np.float32)
    for c, wt in zip(present, w):
        cw[c] = wt

    return cw


def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    return DataLoader(IDSDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)


def train_mlp(X_train, y_tr, X_val, y_va, n_classes: int,
              class_weights: np.ndarray, checkpoint_path):
    """Train the [128, 64] MLP with class-weighted CE; returns (model, history)."""
    tr_loader = _loader(X_train, y_tr, shuffle=True)
    va_loader = _loader(X_val, y_va, shuffle=False)
    model = IDSModel(X_train.shape[1], n_classes,
                     hidden_sizes=[128, 64], dropout=0.3, activation='relu').to(device)
    model, history = train_model(model, tr_loader, va_loader,
                                 torch.tensor(class_weights), N_EPOCHS, PATIENCE, LR,
                                 device, checkpoint_path=checkpoint_path,
                                 optimizer_name='adam')

    return model, history


def mlp_logits(model, X: np.ndarray, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Raw (pre-softmax) logits for an array, batched; used for calibration."""
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.tensor(X[i:i + batch_size]).to(device)
            out.append(model(xb).cpu().numpy())

    return np.concatenate(out)


def train_rf(X_train, y_tr, seed: int = SEED) -> RandomForestClassifier:
    rf = RandomForestClassifier(n_estimators=200, max_depth=20,
                                class_weight='balanced', n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_tr)

    return rf


def train_xgb(X_train, y_tr, n_classes: int, seed: int = SEED) -> xgb.XGBClassifier:
    present = np.unique(y_tr)
    w = compute_class_weight('balanced', classes=present, y=y_tr)
    sample_weight = np.ones(len(y_tr), dtype=np.float32)

    for c, wt in zip(present, w):
        sample_weight[y_tr == c] = wt

    clf = xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                            tree_method='hist', n_jobs=-1, random_state=seed,
                            objective='binary:logistic' if n_classes == 2 else 'multi:softprob',
                            eval_metric='logloss' if n_classes == 2 else 'mlogloss')
    clf.fit(X_train, y_tr, sample_weight=sample_weight)

    return clf
