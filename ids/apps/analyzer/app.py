"""RT-IDS backend — FastAPI REST endpoint for the React frontend.

Run:
    python -m ids.apps.analyzer.app

REST API:   http://localhost:7860/api/classify
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from ids.core.config import MODELS_DIR
from ids.core.timing import Timer
from ids.runtime.predictor import MLPClassifier, RFClassifier


def _parse(stem: str) -> tuple[str, str] | None:
    """('random', '2') from 'ids_dnn_random_2class' (split may contain underscores)."""
    parts = stem.split('_')
    if len(parts) < 4 or not parts[-1].endswith('class'):
        return None
    return '_'.join(parts[2:-1]), parts[-1].removesuffix('class')


def discover_predictors() -> dict:
    """Load all trained models from disk, keyed 'model_type/split/mode'."""
    found: dict = {}
    for ckpt in MODELS_DIR.glob('ids_dnn_*_*class.pth'):
        sm = _parse(ckpt.stem)
        if sm is None:
            continue
        split, mode = sm
        try:
            found[f'mlp/{split}/{mode}'] = MLPClassifier(MODELS_DIR, split=split, mode=mode)
        except FileNotFoundError:
            continue
    for kind in ('rf',):
        for ckpt in MODELS_DIR.glob(f'ids_{kind}_*_*class.joblib'):
            sm = _parse(ckpt.stem)
            if sm is None:
                continue
            split, mode = sm
            try:
                found[f'{kind}/{split}/{mode}'] = RFClassifier(MODELS_DIR, kind, split=split, mode=mode)
            except FileNotFoundError:
                continue
    return found


PREDICTORS = discover_predictors()
if not PREDICTORS:
    raise SystemExit(
        f'No trained models found in {MODELS_DIR}. '
        'Run the notebook end-to-end before launching the demo.'
    )


EXPLAIN_KEY = 'mlp/random/8'
EXPLAINER = None
if EXPLAIN_KEY in PREDICTORS:
    try:
        from ids.runtime.explain import FlowExplainer
        from ids.data.sampler import FlowSampler
        EXPLAINER = FlowExplainer(PREDICTORS[EXPLAIN_KEY], FlowSampler())
    except Exception as e:
        print(f'[app] SHAP explainer unavailable, classify will omit top_features: {e}', flush=True)
        EXPLAINER = None


def _aggregate(pred: dict) -> dict:
    """Aggregate per-flow predictions into a single result."""
    probs    = pred['probabilities']
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


def _explain_dominant(predictor, predictor_key: str, df: pl.DataFrame, pred: dict,
                      top_label: str, top_k: int = 8) -> list[dict] | None:
    """SHAP explanation for the dominant prediction; returns None if unavailable."""
    if EXPLAINER is None or predictor_key != EXPLAIN_KEY:
        return None
    try:
        class_names = pred['class_names']
        top_idx = class_names.index(top_label)
        probs = np.asarray(pred['probabilities'])
        rep = int(probs[:, top_idx].argmax())
        x_scaled = predictor.preprocess(df)[rep]
        reasons = EXPLAINER.explain(x_scaled, df.row(rep, named=True), top_idx, top_k=top_k)
        return [{'feature': r['feature'], 'contribution': r['shap']} for r in reasons]
    except Exception as e:
        print(f'[app] SHAP explanation skipped: {e}', flush=True)
        return None


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
    split:      str        = Form('random'),
):
    predictor_key = f'{model_type}/{split}/{mode}'
    if predictor_key not in PREDICTORS:
        available = list(PREDICTORS.keys())
        return {'error': f'Model {predictor_key!r} not found. Available: {available}'}

    predictor = PREDICTORS[predictor_key]
    timer = Timer()
    t0 = time.perf_counter()

    with timer.span('read_ms'):
        contents = await file.read()

    fname = file.filename or 'upload.csv'
    is_pcap = fname.lower().endswith(('.pcap', '.pcapng', '.cap'))
    with timer.span('extract_ms'):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / fname
            path.write_bytes(contents)
            if is_pcap:
                from ids.runtime.extractor import extract_features
                df = extract_features(str(path))
            else:
                df = pl.read_csv(str(path))

    if df.height == 0:
        return {'error': 'No flows could be extracted from the upload. For packet '
                         'captures, ensure the file contains IPv4 TCP/UDP traffic.'}

    # predict() records preprocess_ms + inference_ms (pure model call) into the same timer.
    pred = predictor.predict(df, timer=timer)
    with timer.span('aggregate_ms'):
        result = _aggregate(pred)

    with timer.span('explain_ms'):
        top_features = _explain_dominant(predictor, predictor_key, df, pred, result['top_label'])
    if top_features is not None:
        result['top_features'] = top_features

    total_ms = (time.perf_counter() - t0) * 1000
    spans = timer.as_dict()
    # "overhead" = everything that is not the pure model call, so the demo can show
    # how little of the wall time the inference itself costs.
    spans['inference_per_flow_ms'] = round(spans.get('inference_ms', 0.0) / max(df.height, 1), 4)
    spans['total_server_ms'] = round(total_ms, 3)
    result['timing'] = spans
    result['processing_time_ms'] = round(total_ms)
    result['model_type']  = model_type
    result['mode']        = mode
    result['split']       = split
    result['file_name']   = file.filename
    result['input_type']  = 'pcap' if is_pcap else 'csv'
    result['success']     = True

    return result


if __name__ == '__main__':
    uvicorn.run(api, host='0.0.0.0', port=7860)
