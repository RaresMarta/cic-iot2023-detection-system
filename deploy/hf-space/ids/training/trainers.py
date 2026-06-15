"""Model training: MLP (PyTorch) and the Random Forest tree baseline.

Hyperparameters are read per ``(model, mode)`` from hparams.json (the single
source of truth, written by ``ids.training.tune``) — never hardcoded here; see
the thesis methodology chapter.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from ids.core.config import BATCH_SIZE, N_EPOCHS, PATIENCE, SEED, load_hparams
from ids.core.models import IDSDataset, IDSModel, train_model, device


def balanced_class_weights(y_tr: np.ndarray, n_classes: int) -> np.ndarray:
    """Balanced-frequency class weights, length ``n_classes`` (absent classes -> 1)."""
    present = np.unique(y_tr)
    w = compute_class_weight('balanced', classes=present, y=y_tr)
    cw = np.ones(n_classes, dtype=np.float32)
    for c, wt in zip(present, w):
        cw[c] = wt

    return cw


def _loader(X: np.ndarray, y: np.ndarray, shuffle: bool,
            batch_size: int = BATCH_SIZE) -> DataLoader:
    return DataLoader(IDSDataset(torch.tensor(X), torch.tensor(y)),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_mlp(X_train, y_tr, X_val, y_va, n_classes: int,
              class_weights: np.ndarray, checkpoint_path, mode):
    """Train the MLP with class-weighted CE. The architecture and optimizer
    hyperparameters for this ``mode`` are read from hparams.json (the single
    source of truth written by ``ids.training.tune``); returns (model, history)."""
    hp = load_hparams('mlp', mode)
    tr_loader = _loader(X_train, y_tr, shuffle=True, batch_size=hp['batch_size'])
    va_loader = _loader(X_val, y_va, shuffle=False, batch_size=hp['batch_size'])
    model = IDSModel(X_train.shape[1], n_classes,
                     hidden_sizes=hp['hidden'], dropout=hp['dropout'],
                     activation=hp['activation']).to(device)
    model, history = train_model(model, tr_loader, va_loader,
                                 torch.tensor(class_weights), N_EPOCHS, PATIENCE, hp['lr'],
                                 device, checkpoint_path=checkpoint_path,
                                 optimizer_name=hp['optimizer'])

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


def train_rf(X_train, y_tr, mode, seed: int = SEED) -> RandomForestClassifier:
    """Random Forest; hyperparameters for this ``mode`` are read from hparams.json
    (written by ``ids.training.tune``). ``class_weight='balanced'`` is held fixed to
    match the project's class-weighted-only imbalance decision. max_depth is capped
    at save time (see ``tune.RF_MAX_DEPTH_CAP``) to keep the serialized forest
    tractable --- the searched depth ~33 produces a >1 GB model on full train."""
    hp = load_hparams('rf', mode)
    rf = RandomForestClassifier(n_estimators=hp['n_estimators'], max_depth=hp['max_depth'],
                                min_samples_split=hp['min_samples_split'],
                                min_samples_leaf=hp['min_samples_leaf'],
                                max_features=hp['max_features'], class_weight='balanced',
                                n_jobs=-1, random_state=seed)
    rf.fit(X_train, y_tr)

    return rf
