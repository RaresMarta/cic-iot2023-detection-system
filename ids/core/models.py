"""PyTorch model, dataset, and training utilities."""

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)

device = torch.device(
    'cuda' if torch.cuda.is_available()
    else ('mps' if torch.backends.mps.is_available() else 'cpu')
)

_ACTIVATIONS = {
    'relu':      nn.ReLU,
    'elu':       nn.ELU,
    'leakyrelu': nn.LeakyReLU,
}


class IDSDataset(Dataset):
    """PyTorch Dataset wrapper for (X, y) arrays."""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class IDSModel(nn.Module):
    """Variable-depth MLP with BatchNorm and Dropout per layer."""

    def __init__(self, n_features: int, n_classes: int,
                 hidden_sizes: list | None = None,
                 dropout: float = 0.3,
                 activation: str = 'relu'):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]
        Act = _ACTIVATIONS[activation]
        layers = []
        in_size = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(in_size, h), nn.BatchNorm1d(h), Act(), nn.Dropout(dropout)]
            in_size = h
        layers.append(nn.Linear(in_size, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_model(model: nn.Module, train_loader, val_loader,
                class_weights: torch.Tensor, n_epochs: int, patience: int,
                lr: float, device: torch.device, checkpoint_path=None,
                optimizer_name: str = 'adam', run=None, trial=None):
    """Train with class-weighted CE, ReduceLROnPlateau, early stopping on val loss.

    Args:
        optimizer_name: 'adam' or 'adamw'
        trial: optional Optuna trial for pruning
    Returns:
        (trained_model, history_dict)
    """
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    opt_cls = torch.optim.AdamW if optimizer_name == 'adamw' else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )
    use_amp = device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(n_epochs):
        model.train()
        running = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast('cuda', enabled=use_amp):
                loss = criterion(model(Xb), yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * Xb.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                with autocast('cuda', enabled=use_amp):
                    loss = criterion(model(Xb), yb)
                running += loss.item() * Xb.size(0)
        val_loss = running / len(val_loader.dataset)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        print(f'  Epoch {epoch+1:02d} — train: {train_loss:.4f} — val: {val_loss:.4f} — lr: {current_lr:.2e}')
        if run is not None:
            run.log({'train_loss': train_loss, 'val_loss': val_loss, 'lr': current_lr}, step=epoch + 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                torch.save(best_state, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'  Early stop @ epoch {epoch+1}')
                break

        # Optuna pruning — kill unpromising trials early
        if trial is not None:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise __import__('optuna').exceptions.TrialPruned()

    if best_state is not None:  # set on the first epoch; guards the type checker
        model.load_state_dict(best_state)

    return model, history


def evaluate(model: nn.Module, test_loader, class_names: list,
             device: torch.device):
    """Compute metrics and confusion matrix on test set."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            out = model(Xb.to(device, non_blocking=True))
            preds.append(out.argmax(dim=1).cpu().numpy())
            labels.append(yb.numpy())
    y_pred = np.concatenate(preds)
    y_true = np.concatenate(labels)

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'report': classification_report(y_true, y_pred, target_names=class_names, zero_division=0, digits=4),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'y_true': y_true,
        'y_pred': y_pred,
    }
