"""Optuna hyperparameter search (MLP + Random Forest); random-split headline.

This module is the ONLY writer of hparams.json — the single source of truth that
``ids.training.trainers`` and serving read back. ``main()`` runs the search for a
``(model, mode)`` cell and persists the normalized best params via ``save_hparams``."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import optuna
from optuna.samplers import TPESampler

from ids.core.config import N_FEATURES, HPARAMS_PATH
from ids.core.models import IDSDataset, IDSModel, train_model, evaluate
from ids.data.preprocessing import fit_preprocess

RF_MAX_DEPTH_CAP = 25


def _canonical_mlp(best: dict) -> dict:
    """Optuna best_params (flat ``hidden_0/1/...``) -> the schema trainers read."""
    return {
        'hidden': [best[f'hidden_{i}'] for i in range(best['n_layers'])],
        'dropout': best['dropout'],
        'activation': best['activation'],
        'optimizer': best['optimizer'],
        'lr': best['lr'],
        'batch_size': best['batch_size'],
    }


def _canonical_rf(best: dict) -> dict:
    """Optuna best_params -> the schema trainers read, applying the depth cap."""
    depth = min(best['max_depth'], RF_MAX_DEPTH_CAP)
    if best['max_depth'] > RF_MAX_DEPTH_CAP:
        print(f"  capping max_depth {best['max_depth']} -> {RF_MAX_DEPTH_CAP} "
              f"(serialized-forest size guard)")
    return {
        'n_estimators': best['n_estimators'],
        'max_depth': depth,
        'min_samples_split': best['min_samples_split'],
        'min_samples_leaf': best['min_samples_leaf'],
        'max_features': best['max_features'],
    }


def save_hparams(model: str, mode: str, params: dict) -> None:
    """Write the tuned ``params`` into the ``(model, mode)`` cell of hparams.json,
    leaving every other cell untouched. The only writer of that file."""
    data = json.loads(HPARAMS_PATH.read_text()) if HPARAMS_PATH.exists() else {}
    data.setdefault(model, {})[str(mode)] = params
    HPARAMS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(f'saved {model}/{mode} -> {HPARAMS_PATH}')


def _log_optuna_plots(run, study) -> None:
    """Log Optuna study visualisations (optimization history + param importances)
    to the active wandb run. No-op if plotly/optuna.visualization is unavailable
    or there are too few completed trials."""
    try:
        import optuna.visualization as ov
        run.log({"optuna/optimization_history": ov.plot_optimization_history(study),
                 "optuna/param_importances": ov.plot_param_importances(study)})
    except Exception as e:
        print(f"  [wandb] optuna plots skipped: {e}")


def run_hpo(
    X_all: np.ndarray,
    y_all_34: np.ndarray,
    split_indices: dict,
    remap_labels: Callable,
    device: torch.device,
    seed: int = 42,
    n_trials: int = 15,
    tune_epochs: int = 10,
    tune_split: str = "random",
    tune_mode: str = "8",
    subsample: int = 100_000,
    wandb_enabled: bool = False,
) -> dict:
    """Optuna TPE search for the MLP; maximises validation macro-F1.

    Mode-general (encoder fit on the remapped labels, like run_training), so
    ``tune_mode='8'`` works. Objective is the trained model's val macro-F1 (the
    imbalance-sensitive metric the thesis reports), not val loss."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tr_full, va, te = split_indices[tune_split]
    rng = np.random.default_rng(seed)
    tr  = rng.choice(tr_full, size=min(subsample, len(tr_full)), replace=False)

    X_tr, X_va, _, _ = fit_preprocess(X_all, tr, va, te)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    del X_tr, X_va

    le = LabelEncoder().fit(remap_labels(y_all_34, tune_mode))
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
        model = IDSModel(X_tr_t.shape[1], n_cls, hidden_sizes=hidden, dropout=dropout, activation=act).to(device)

        model, _ = train_model(
            model, train_loader, val_loader, weights_t,
            tune_epochs, 5, lr, device,
            optimizer_name=opt_name, trial=None,
        )
        return evaluate(model, val_loader, list(le.classes_), device)["macro_f1"]

    db_path = os.path.join(tempfile.gettempdir(), "optuna_ids.db")
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        storage=f"sqlite:///{db_path}",
        study_name="ids-mlp-hpo-f1",
        load_if_exists=True,
    )
    study.optimize(_objective, n_trials=n_trials, n_jobs=2, show_progress_bar=True)

    del X_tr_t, X_va_t

    best = study.best_params
    print(f"Best val macro-F1 : {study.best_value:.4f}")
    for k, v in best.items():
        print(f"  {k}: {v}")

    if wandb_enabled:
        import wandb
        run = wandb.init(project="cic-iot2023-ids", name="optuna-mlp-best-trial",
                         job_type="hparam-search")
        run.log({"best_val_macro_f1": study.best_value, **best})
        _log_optuna_plots(run, study)
        wandb.finish()

    return best


def run_hpo_rf(
    X_all: np.ndarray,
    y_all_34: np.ndarray,
    split_indices: dict,
    remap_labels: Callable,
    seed: int = 42,
    n_trials: int = 30,
    tune_split: str = "random",
    tune_mode: str = "8",
    subsample: int = 100_000,
    wandb_enabled: bool = False,
) -> dict:
    """Optuna TPE search for the Random Forest; maximises validation macro-F1.

    Mirrors ``run_hpo`` (MLP) but for the tree baseline. RF has no iterative
    ``val_loss`` to minimise, so the objective is validation macro-F1 (the
    imbalance-sensitive metric the thesis reports). Unlike the MLP search this
    is mode-general: the label encoder is fit on ``remap_labels(y_all_34, mode)``
    the same way ``run_training`` does, so ``tune_mode='8'`` works directly.
    ``class_weight='balanced'`` is held fixed to match the project's
    class-weighted-only imbalance decision. The search fits on a subsample of
    train for speed; the final model (in ``trainers.train_rf``) trains on full
    train with the returned params baked in.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    tr_full, va, te = split_indices[tune_split]
    rng = np.random.default_rng(seed)
    tr  = rng.choice(tr_full, size=min(subsample, len(tr_full)), replace=False)

    X_tr, X_va, _, _ = fit_preprocess(X_all, tr, va, te)

    le = LabelEncoder().fit(remap_labels(y_all_34, tune_mode))
    y_tr_enc = le.transform(remap_labels(y_all_34[tr], tune_mode))
    y_va_enc = le.transform(remap_labels(y_all_34[va], tune_mode))

    def _objective(trial: optuna.Trial) -> float:
        rf = RandomForestClassifier(
            n_estimators=trial.suggest_int("n_estimators", 100, 600, step=50),
            max_depth=trial.suggest_int("max_depth", 10, 40),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, 0.7]),
            class_weight="balanced", n_jobs=-1, random_state=seed,
        )
        rf.fit(X_tr, y_tr_enc)
        return f1_score(y_va_enc, rf.predict(X_va), average="macro")

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        study_name="ids-rf-hparam-search",
    )
    study.optimize(_objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    best = study.best_params
    print(f"Best val macro-F1 : {study.best_value:.4f}")
    for k, v in best.items():
        print(f"  {k}: {v}")

    if wandb_enabled:
        import wandb
        run = wandb.init(project="cic-iot2023-ids", name="optuna-rf-best-trial",
                         job_type="hparam-search")
        run.log({"best_val_macro_f1": study.best_value, **best})
        _log_optuna_plots(run, study)
        wandb.finish()

    return best


def main() -> None:
    """Headless HPO entry point: ``python -m ids.training.tune --model mlp --mode 2``.

    Loads the dataset, builds the requested split, runs the search, and writes the
    normalized best params into the ``(model, mode)`` cell of hparams.json (the
    single source of truth that training and serving read back)."""
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m ids.training.tune",
        description="Optuna search; prints best params to bake into trainers.py.")
    p.add_argument("--model", choices=["rf", "mlp"], default="rf",
                   help="model to tune")
    p.add_argument("--split", default="random",
                   choices=["temporal", "per_csv", "random"])
    p.add_argument("--mode", default="8", choices=["2", "8", "34"])
    p.add_argument("--trials", type=int, default=30)
    p.add_argument("--subsample", type=int, default=100_000)
    p.add_argument("--wandb", action="store_true")
    args = p.parse_args()

    from ids.core.config import SEED
    from ids.core.labels import remap_labels
    from ids.data.preprocessing import SPLIT_FUNCS
    from ids.training.data import load_dataset

    X_all, y_all_34, source_csv = load_dataset()
    tr, va, te = SPLIT_FUNCS[args.split](y_all_34, source_csv, SEED)
    split_indices = {args.split: (tr, va, te)}

    if args.model == "rf":
        best = run_hpo_rf(X_all, y_all_34, split_indices, remap_labels,
                          seed=SEED, n_trials=args.trials, tune_split=args.split,
                          tune_mode=args.mode, subsample=args.subsample,
                          wandb_enabled=args.wandb)
        params = _canonical_rf(best)
    else:
        from ids.core.models import device
        best = run_hpo(X_all, y_all_34, split_indices, remap_labels, device,
                       seed=SEED, n_trials=args.trials, tune_split=args.split,
                       tune_mode=args.mode, subsample=args.subsample,
                       wandb_enabled=args.wandb)
        params = _canonical_mlp(best)

    save_hparams(args.model, args.mode, params)
    print("\nBEST PARAMS:", params)


if __name__ == "__main__":
    main()
