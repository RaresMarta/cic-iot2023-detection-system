"""Headless reproduction entry point: ``python -m training [options]``.

Retrains every (split, mode) combination, persists all serving and reporting
artifacts, and prints the paper-ready numbers.
"""
from __future__ import annotations

import argparse

from ids.training import run_training
from ids.training.plots import MODEL_LABELS


def main() -> None:
    p = argparse.ArgumentParser(
        prog='python -m training',
        description='Retrain MLP + RF and regenerate all artifacts, '
                    'the metrics cache, calibration, and the paper numbers.')
    p.add_argument('--splits', nargs='+', default=['temporal'],
                   choices=['temporal', 'per_csv', 'random'],
                   help='splits to run (first one is the headline: its calibration '
                        'artifact is the one served)')
    p.add_argument('--modes', nargs='+', default=['2', '8'],
                   choices=['2', '8', '34'], help='classification granularities')
    p.add_argument('--wandb', action='store_true',
                   help='log per-model test metrics to Weights & Biases')
    args = p.parse_args()

    out = run_training(splits=args.splits, modes=args.modes,
                       wandb_enabled=args.wandb)

    print('\n================ PAPER NUMBERS ================')
    for split, R in out['splits'].items():
        for mode in args.modes:
            print(f'\n--- {split} {mode}-class TEST ---')
            print(f'{"model":<14}{"acc":>8}{"wF1":>8}{"macroF1":>9}{"valW-F1":>9}{"valMacro":>9}')
            for m in ('mlp', 'rf'):
                t = R[(f'mode{mode}', m, 'test')]
                v = R[(f'mode{mode}', m, 'val')]
                print(f'{MODEL_LABELS[m]:<14}{t["accuracy"]:>8.4f}{t["weighted_f1"]:>8.4f}'
                      f'{t["macro_f1"]:>9.4f}{v["weighted_f1"]:>9.4f}{v["macro_f1"]:>9.4f}')


if __name__ == '__main__':
    main()
