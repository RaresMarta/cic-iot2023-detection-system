"""Extract one representative HELD-OUT flow per 8-class family for the analyzer's
manual-entry preset loaders.

Rows are taken ONLY from the test split (the same SPLIT_FUNCS['random'] + SEED used
in training), so every preset is a flow the model never trained on — "this is a real
flow the model has not seen" is a true claim.

Selection: for each family, score every held-out row of that family through the REAL
inference pipeline (MLP, 8-class — the demo's primary path) and keep the row the model
classifies *correctly as its own family with the highest confidence*. This guarantees
each preset actually predicts its desired label end-to-end. (The earlier "row closest
to the per-feature median in raw space" heuristic landed flows near inter-class
boundaries — raw-space L1 distance is dominated by huge-magnitude features like Tot sum
/ Rate — so several presets mispredicted through the validated pipeline even though the
model is accurate overall.)

IMPORTANT: run this only against the preprocessor that is actually deployed and that
reproduces the recorded test accuracy. A mismatched preprocessor will pick presets that
don't generalize to the live backend.

Run:  .venv/bin/python scripts/extract_flow_presets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ids.core.config import SEED, X_COLUMNS_SELECTED
from ids.core.labels import remap_labels
from ids.data.preprocessing import SPLIT_FUNCS
from ids.runtime.predictor import MLPClassifier
from ids.training.data import load_dataset

MODELS_DIR = Path(__file__).resolve().parents[1] / 'models'
OUT = (Path(__file__).resolve().parents[2]
       / 'ids-frontend' / 'src' / 'app' / 'data' / 'flow_presets.json')


def main() -> None:
    cols = X_COLUMNS_SELECTED
    X_all, y34, src = load_dataset()
    _, _, te = SPLIT_FUNCS['random'](y34, src, SEED)   # held-out test indices only
    X = X_all[te].astype(float)
    fam8 = remap_labels(y34[te], '8')

    predictor = MLPClassifier(MODELS_DIR, split='random', mode='8')

    presets: dict[str, dict[str, float]] = {}
    for fam in sorted(set(fam8)):
        idx = np.where(fam8 == fam)[0]
        Xf = X[idx]
        med = np.nanmedian(np.where(np.isfinite(Xf), Xf, np.nan), axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xclean = np.where(np.isfinite(Xf), Xf, med)

        df = pl.DataFrame({c: Xclean[:, j] for j, c in enumerate(cols)})
        out = predictor.predict(df)
        labels = np.asarray(out['labels'])
        conf = np.asarray(out['confidences'])

        correct = labels == fam
        if correct.any():
            cand = np.where(correct)[0]
            pick = cand[int(np.argmax(conf[cand]))]
            status = f'OK   conf={conf[pick]:.3f}'
        else:
            pick = int(np.argmax(conf))
            status = f'!!   NO correct row; best={labels[pick]} conf={conf[pick]:.3f}'

        row = Xclean[pick]
        presets[str(fam)] = {c: round(float(v), 6) for c, v in zip(cols, row)}
        print(f'{fam:11} test-n={len(idx):>8,}  {status}')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(presets, indent=2) + '\n')
    print(f'wrote {OUT} ({len(presets)} families x {len(cols)} features, test-split only)')


if __name__ == '__main__':
    main()
