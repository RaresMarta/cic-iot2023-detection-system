"""Generate a sample CSV that the served model actually classifies as Benign.

The existing sample_benign_browsing.csv aggregates to DDoS under RF/8-class, so the
demo never shows the benign banner. This pulls real Benign rows from the dataset,
runs them through the SAME predictor the API uses, keeps only flows the model
confidently calls Benign, and confirms the page-level aggregate (mean-prob argmax,
matching app._aggregate) is Benign before writing the file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ids.runtime.predictor import RFClassifier, MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / 'models'
PARQUET = ROOT / 'data' / 'cic_iot_2023.parquet'
OUT = ROOT / 'data' / 'samples' / 'sample_benign_browsing.csv'

# Match the existing sample's column order exactly (the 39 CIC features, no Label).
SAMPLE_COLS = pl.read_csv(OUT).columns

CANDIDATE_POOL = 8000   # benign rows to score
TARGET_ROWS = 80        # final sample size (same as the old benign sample)
# Benign is genuinely hard here (Web/Benign overlap), so per-flow confidence is
# modest (~0.43 mean). We don't gate on a confidence floor — we require BOTH
# served models to predict Benign, then rank by combined benign probability and
# take the strongest. The page verdict is the mean-prob argmax, which clears
# Benign comfortably once the kept flows all lean benign.


def aggregate_label(probs: np.ndarray, class_names: list[str]) -> tuple[str, float]:
    mean_p = probs.mean(axis=0)
    idx = int(mean_p.argmax())
    return class_names[idx], float(mean_p[idx])


def main() -> None:
    benign = (
        pl.scan_parquet(PARQUET)
        .filter(pl.col('Label') == 'Benign_Final')
        .select(SAMPLE_COLS)
        .head(CANDIDATE_POOL)
        .collect()
    )
    print(f'loaded {benign.height} benign candidate rows')

    # Score through BOTH served 8-class backends so the sample is benign regardless
    # of which model the demo picks.
    rf = RFClassifier(MODELS_DIR, kind='rf', split='random', mode='8')
    mlp = MLPClassifier(MODELS_DIR, split='random', mode='8')

    rf_pred = rf.predict(benign)
    mlp_pred = mlp.predict(benign)

    bi_rf = list(rf_pred['class_names']).index('Benign')
    bi_mlp = list(mlp_pred['class_names']).index('Benign')

    keep_mask = (np.asarray(rf_pred['labels']) == 'Benign') & (np.asarray(mlp_pred['labels']) == 'Benign')

    n_keep = int(keep_mask.sum())
    print(f'flows predicted Benign by BOTH models: {n_keep}')
    if n_keep < TARGET_ROWS:
        raise SystemExit(f'only {n_keep} qualifying rows; raise CANDIDATE_POOL')

    # Of the agreed-benign flows, take the strongest by combined benign probability.
    combined = rf_pred['probabilities'][:, bi_rf] + mlp_pred['probabilities'][:, bi_mlp]
    agreed_idx = np.where(keep_mask)[0]
    keep_idx = agreed_idx[np.argsort(combined[agreed_idx])[::-1][:TARGET_ROWS]]
    sample = benign[sorted(keep_idx.tolist())]

    # Verify the page-level verdict (mean-prob argmax) is Benign for both backends.
    for name, pred in (('RF', rf), ('MLP', mlp)):
        p = pred.predict(sample)
        lbl, conf = aggregate_label(p['probabilities'], list(p['class_names']))
        print(f'{name} aggregate verdict: {lbl} ({conf:.1%})')
        if lbl != 'Benign':
            raise SystemExit(f'{name} aggregate is {lbl}, not Benign — aborting')

    sample.write_csv(OUT)
    print(f'wrote {sample.height} rows -> {OUT}')


if __name__ == '__main__':
    main()
