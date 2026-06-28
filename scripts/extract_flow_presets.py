"""Extract one representative HELD-OUT flow per 8-class family for the analyzer's
manual-entry preset loaders.

Rows are taken ONLY from the test split (the same SPLIT_FUNCS['random'] + SEED used
in training), so every preset is a flow the model never trained on — "this is a real
flow the model has not seen" is a true claim. For each family it picks the actual
test row closest (L1) to that family's per-feature median: a real flow, not a
synthetic average, but representative rather than a noisy outlier.

Run:  python scripts/extract_flow_presets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids.core.config import SEED, X_COLUMNS_SELECTED
from ids.core.labels import remap_labels
from ids.data.preprocessing import SPLIT_FUNCS
from ids.training.data import load_dataset

OUT = (Path(__file__).resolve().parents[2]
       / 'ids-frontend' / 'src' / 'app' / 'data' / 'flow_presets.json')


def main() -> None:
    cols = X_COLUMNS_SELECTED
    X_all, y34, src = load_dataset()
    _, _, te = SPLIT_FUNCS['random'](y34, src, SEED)   # held-out test indices only
    X = X_all[te].astype(float)
    fam8 = remap_labels(y34[te], '8')

    presets: dict[str, dict[str, float]] = {}
    for fam in sorted(set(fam8)):
        Xf = X[fam8 == fam]
        med = np.nanmedian(Xf, axis=0)
        clean = np.where(np.isfinite(Xf), Xf, np.nan)
        row = Xf[int(np.nanargmin(np.nansum(np.abs(clean - med), axis=1)))]
        presets[str(fam)] = {
            c: round(float(v if np.isfinite(v) else (med[j] if np.isfinite(med[j]) else 0.0)), 6)
            for j, (c, v) in enumerate(zip(cols, row))
        }
        print(f'{fam:11} test-n={int((fam8 == fam).sum()):>8,}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(presets, indent=2) + '\n')
    print(f'wrote {OUT} ({len(presets)} families x {len(cols)} features, test-split only)')


if __name__ == '__main__':
    main()
