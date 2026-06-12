"""RT-IDS backend — FastAPI REST endpoint for the React frontend.

Run:
    python -m demo.app

REST API:   http://localhost:7860/api/classify
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import polars as pl
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from demo.inference import IDSPredictor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR   = PROJECT_ROOT / 'models'


def discover_predictors() -> dict[str, IDSPredictor]:
    found: dict[str, IDSPredictor] = {}
    for ckpt in MODELS_DIR.glob('ids_dnn_*_*class.pth'):
        parts = ckpt.stem.split('_')
        if len(parts) < 4 or not parts[-1].endswith('class'):
            continue
        mode  = parts[-1].removesuffix('class')
        split = '_'.join(parts[2:-1])
        try:
            found[f'{split} / {mode}-class'] = IDSPredictor(MODELS_DIR, split=split, mode=mode)
        except FileNotFoundError:
            continue
    return found


PREDICTORS = discover_predictors()
if not PREDICTORS:
    raise SystemExit(
        f'No trained models found in {MODELS_DIR}. '
        'Run the notebook end-to-end before launching the demo.'
    )


def _aggregate(pred: dict) -> dict:
    """Aggregate per-flow predictions into a single result."""
    probs    = pred['probabilities']          # (n_flows, n_classes)
    mean_p   = probs.mean(axis=0)
    top_idx  = mean_p.argmax()
    top_label = pred['class_names'][top_idx]

    labels = pred['labels']
    breakdown = {}
    for lbl in labels:
        breakdown[str(lbl)] = breakdown.get(str(lbl), 0) + 1

    return {
        'top_label':    top_label,
        'confidence':   float(mean_p[top_idx]),
        'probabilities': {name: float(p) for name, p in zip(pred['class_names'], mean_p)},
        'class_names':  pred['class_names'],
        'flow_count':   len(labels),
        'breakdown':    breakdown,
    }


def classify_csv(csv_path: Path, predictor_key: str):
    df  = pl.read_csv(str(csv_path))
    return PREDICTORS[predictor_key].predict(df)


# ── FastAPI app ────────────────────────────────────────────────────────────
api = FastAPI(title='RT-IDS API')
api.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@api.get('/api/health')
def health():
    return {'status': 'ok', 'models': list(PREDICTORS.keys())}


@api.post('/api/classify')
async def classify_endpoint(
    file:       UploadFile = File(...),
    model_type: str        = Form('mlp'),
    mode:       str        = Form('2'),
    split:      str        = Form('temporal'),
):
    predictor_key = f'{split} / {mode}-class'
    if predictor_key not in PREDICTORS:
        available = list(PREDICTORS.keys())
        return {'error': f'Model {predictor_key!r} not found. Available: {available}'}

    t0 = time.time()
    contents = await file.read()

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / (file.filename or 'upload.csv')
        csv_path.write_bytes(contents)
        pred = classify_csv(csv_path, predictor_key)

    result = _aggregate(pred)
    result['processing_time_ms'] = round((time.time() - t0) * 1000)
    result['model_type']  = model_type
    result['mode']        = mode
    result['split']       = split
    result['file_name']   = file.filename
    result['success']     = True

    return result


if __name__ == '__main__':
    uvicorn.run(api, host='0.0.0.0', port=7860)
