"""Optuna hyperparameter search for the MLP on the temporal/2-class split."""
from __future__ import annotations

import os
import tempfile
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import wandb

from config import N_FEATURES
from models import IDSDataset, IDSModel, train_model
from preprocessing import fit_preprocess


def run_hpo(
    X_all: np.ndarray,
    y_all_34: np.ndarray,
    split_indices: dict,
    remap_labels: Callable,
    label_dict_2class: dict,
    device: torch.device,
    seed: int = 42,
    n_trials: int = 15,
    tune_epochs: int = 10,
    tune_split: str = "temporal",
    tune_mode: str = "2",
    subsample: int = 100_000,
) -> dict:
    """Run Optuna TPE search and return the best hyperparameter dict."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tr_full, va, te = split_indices[tune_split]
    rng = np.random.default_rng(seed)
    tr  = rng.choice(tr_full, size=min(subsample, len(tr_full)), replace=False)

    X_tr, X_va, _, _, _ = fit_preprocess(X_all, tr, va, te)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    del X_tr, X_va

    le = LabelEncoder().fit(sorted(set(label_dict_2class.values())))
    y_tr_enc = le.transform(remap_labels(y_all_34[tr], tune_mode))
    y_va_enc = le.transform(remap_labels(y_all_34[va], tune_mode))
    n_cls    = len(le.classes_)

    present = np.unique(y_tr_enc)
    w       = compute_class_weight("balanced", classes=present, y=y_tr_enc)
    wfull   = np.ones(n_cls, dtype=np.float32)
    wfull[present] = w
    weights_t = torch.tensor(wfull, dtype=torch.float32)

    def _objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 2, 4)
        hidden   = [trial.suggest_int(f"hidden_{i}", 64, 512, step=32)
                    for i in range(n_layers)]
        dropout  = trial.suggest_float("dropout", 0.1, 0.5)
        act      = trial.suggest_categorical("activation", ["relu", "elu", "leakyrelu"])
        lr       = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        opt_name = trial.suggest_categorical("optimizer", ["adam", "adamw"])
        bs       = trial.suggest_categorical("batch_size", [2048, 4096, 8192])

        gen = torch.Generator().manual_seed(seed)
        train_loader = DataLoader(
            IDSDataset(X_tr_t, torch.tensor(y_tr_enc, dtype=torch.long)),
            batch_size=bs, shuffle=True, generator=gen, num_workers=0,
        )
        val_loader = DataLoader(
            IDSDataset(X_va_t, torch.tensor(y_va_enc, dtype=torch.long)),
            batch_size=bs, num_workers=0,
        )
        model = IDSModel(N_FEATURES, n_cls, hidden_sizes=hidden,
                         dropout=dropout, activation=act).to(device)
        _, history = train_model(
            model, train_loader, val_loader, weights_t,
            tune_epochs, 5, lr, device,
            optimizer_name=opt_name, trial=trial,
        )
        return min(history["val_loss"])

    db_path = os.path.join(tempfile.gettempdir(), "optuna_ids.db")
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        storage=f"sqlite:///{db_path}",
        study_name="ids-hparam-search",
        load_if_exists=True,
    )
    study.optimize(_objective, n_trials=n_trials, n_jobs=2, show_progress_bar=True)

    del X_tr_t, X_va_t

    best = study.best_params
    print(f"Best val loss : {study.best_value:.4f}")
    for k, v in best.items():
        print(f"  {k}: {v}")

    run = wandb.init(project="cic-iot2023-ids", name="optuna-best-trial",
                     job_type="hparam-search")
    run.log({"best_val_loss": study.best_value, **best})
    wandb.finish()

    return best
