"""Optional experiment tracking (wandb). No-op by construction when disabled,
so reproducing results never requires a wandb login.

One wandb run per ``run_training`` invocation. It logs, all namespaced by
``{model}/{split}/{mode}class``:
  - per-epoch training curves (train/val loss, val macro-F1, lr)  [MLP]
  - per-model eval: aggregate scalars, confusion matrix, per-class
    F1/precision/recall, and one-vs-rest PR curves
  - a final MLP-vs-RF comparison table + macro/weighted-F1 bar charts

Every public method is wrapped so a wandb/plotting failure can never crash the
training run — logging is auxiliary.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


class Tracker:
    """wandb logger for a single training run; silent no-op when disabled."""

    def __init__(self, enabled: bool = False, project: str = 'cic-iot2023-ids',
                 name: str | None = None):
        self.enabled = enabled
        self._wandb = None
        self.run = None
        if enabled:
            import wandb  # imported lazily so the package works without wandb
            self._wandb = wandb
            self.run = wandb.init(project=project, name=name, job_type='train')

    # ---- per-epoch curves (MLP) ----
    def log_history(self, split: str, mode: str, model: str, history: dict) -> None:
        if not self.enabled:
            return
        try:
            ns = f'{model}/{split}/{mode}class'
            step_key = f'{ns}/epoch'
            # custom x-axis per namespace so each model's curves get their own epoch axis
            self.run.define_metric(step_key)
            self.run.define_metric(f'{ns}/*', step_metric=step_key)
            for i in range(len(history.get('val_loss', []))):
                row = {step_key: i + 1,
                       f'{ns}/train_loss': history['train_loss'][i],
                       f'{ns}/val_loss': history['val_loss'][i],
                       f'{ns}/lr': history['lr'][i]}
                if history.get('val_macro_f1'):
                    row[f'{ns}/val_macro_f1'] = history['val_macro_f1'][i]
                self.run.log(row)
        except Exception as e:  # never let logging crash training
            print(f'  [wandb] log_history skipped for {model}/{split}/{mode}: {e}')

    # ---- per-model evaluation ----
    def log_eval(self, split: str, mode: str, model: str, class_names: list,
                 y_true, y_pred, y_score=None) -> None:
        if not self.enabled:
            return
        try:
            wandb = self._wandb
            ns = f'{model}/{split}/{mode}class'
            labels = list(range(len(class_names)))

            self.run.summary[f'{ns}/test_macro_f1'] = float(
                f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0))
            self.run.summary[f'{ns}/test_weighted_f1'] = float(
                f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0))

            self.run.log({f'{ns}/confusion_matrix': wandb.plot.confusion_matrix(
                y_true=np.asarray(y_true), preds=np.asarray(y_pred), class_names=class_names)})

            f1c = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            pc  = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            rc  = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
            tbl = wandb.Table(columns=['class', 'f1', 'precision', 'recall'])
            for c, a, b, d in zip(class_names, f1c, pc, rc):
                tbl.add_data(c, float(a), float(b), float(d))
            self.run.log({
                f'{ns}/per_class': tbl,
                f'{ns}/per_class_f1': wandb.plot.bar(tbl, 'class', 'f1',
                                                     title=f'{ns} — per-class F1'),
            })

            # one-vs-rest PR curves — the imbalance-appropriate ranking view
            if y_score is not None:
                y_score = np.asarray(y_score)
                if y_score.ndim == 2 and y_score.shape[1] == len(class_names):
                    self.run.log({f'{ns}/pr_curve': wandb.plot.pr_curve(
                        y_true, y_score, labels=class_names)})
        except Exception as e:  # never let logging crash training
            print(f'  [wandb] log_eval skipped for {model}/{split}/{mode}: {e}')

    # ---- final MLP-vs-RF comparison ----
    def log_comparison(self, results_all: dict, modes) -> None:
        if not self.enabled:
            return
        try:
            wandb = self._wandb
            tbl = wandb.Table(columns=['run', 'model', 'split', 'mode',
                                       'macro_f1', 'weighted_f1'])
            for split, R in results_all.items():
                for mode in modes:
                    for m in ('mlp', 'rf'):
                        t = R.get((f'mode{mode}', m, 'test'))
                        if t:
                            tbl.add_data(f'{m}/{split}/{mode}', m, split, str(mode),
                                         float(t['macro_f1']), float(t['weighted_f1']))
            self.run.log({
                'comparison/table': tbl,
                'comparison/macro_f1': wandb.plot.bar(tbl, 'run', 'macro_f1',
                                                      title='Macro-F1 by model/split/mode'),
                'comparison/weighted_f1': wandb.plot.bar(tbl, 'run', 'weighted_f1',
                                                         title='Weighted-F1 by model/split/mode'),
            })
        except Exception as e:  # never let logging crash training
            print(f'  [wandb] log_comparison skipped: {e}')

    def finish(self) -> None:
        if self.enabled and self.run is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass
