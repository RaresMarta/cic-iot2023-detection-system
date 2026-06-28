"""Write real held-out sample CSVs for the analyzer's file-upload demo.

Replaces the old synthetic data/samples/_make_samples.py (which produced
random.uniform fixtures). Each CSV is a handful of ACTUAL test-split rows for one
family — flows the model never trained on — in the 25-column model order, so the
upload path classifies genuine data and "this is real held-out traffic" is true.

Run:  python scripts/make_test_samples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids.core.config import SEED, X_COLUMNS_SELECTED
from ids.core.labels import remap_labels
from ids.data.preprocessing import SPLIT_FUNCS
from ids.training.data import load_dataset

OUT = Path(__file__).resolve().parents[1] / 'data' / 'samples'
N_PER_FILE = 25
# (family, output filename) — one representative file per demo scenario.
FILES = [
    ('Benign', 'sample_benign.csv'),
    ('DDoS', 'sample_ddos.csv'),
    ('Recon', 'sample_recon.csv'),
]


def main() -> None:
    cols = X_COLUMNS_SELECTED
    X_all, y34, src = load_dataset()
    _, _, te = SPLIT_FUNCS['random'](y34, src, SEED)
    X = X_all[te]
    fam8 = remap_labels(y34[te], '8')
    rng = np.random.default_rng(SEED)

    OUT.mkdir(parents=True, exist_ok=True)
    for family, fname in FILES:
        idx = np.where(fam8 == family)[0]
        take = rng.permutation(idx)[:N_PER_FILE]
        df = pl.DataFrame(X[take], schema=cols)
        df.write_csv(OUT / fname)
        print(f'{family:8} -> {fname} ({len(take)} held-out rows)')
    print(f'wrote {len(FILES)} sample CSVs to {OUT}')


if __name__ == '__main__':
    main()
