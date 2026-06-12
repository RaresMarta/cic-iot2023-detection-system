"""Optional experiment tracking (wandb). No-op by construction when disabled,
so reproducing results never requires a wandb login.
"""
from __future__ import annotations


class Tracker:
    """One wandb run per (split, mode, model); silent no-op when disabled."""

    def __init__(self, enabled: bool = False, project: str = 'cic-iot2023-ids'):
        self.enabled = enabled
        self.project = project
        if enabled:
            import wandb  # imported lazily so the package works without wandb
            self._wandb = wandb

    def log_model(self, split: str, mode: str, model_key: str,
                  config: dict, test_metrics: dict) -> None:
        if not self.enabled:
            return
        run = self._wandb.init(
            project=self.project,
            name=f'{split}/{mode}-class/{model_key}',
            job_type='train',
            config={'split': split, 'mode': mode, 'model': model_key, **config},
        )
        run.log({f'test_{k}': v for k, v in test_metrics.items()})
        self._wandb.finish()
