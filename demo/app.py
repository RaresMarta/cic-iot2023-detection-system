"""RT-IDS backend — Gradio UI + FastAPI REST endpoint for the React frontend.

Run:
    python -m demo.app

Gradio UI:  http://localhost:7860
REST API:   http://localhost:7860/api/classify
"""
from __future__ import annotations

import asyncio
import collections
import json
import random
import tempfile
import time
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import polars as pl
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from demo.cicflowmeter_runner import CICFlowMeterError, run_cicflowmeter
from demo.explain import FlowExplainer
from demo.inference import IDSPredictor
from demo.sampler import FlowSampler

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

# ── Live demo: sampler + SHAP explainer over the 8-class model ───────────────
# The SOC dashboard streams real flows sampled from the training parquet (simulate mode),
# classifies them with the 8-class model, and explains each verdict with SHAP.
_STREAM_KEY = 'temporal / 8-class'
STREAM_PREDICTOR = PREDICTORS.get(_STREAM_KEY) or next(iter(PREDICTORS.values()))
SAMPLER = FlowSampler()
EXPLAINER = FlowExplainer(STREAM_PREDICTOR, SAMPLER)

# In-memory queue of attack families to inject into the live stream (filled by /api/inject).
INJECT_QUEUE: collections.deque[str] = collections.deque()


def _synthetic_endpoints(is_attack: bool) -> tuple[str, str]:
    """Plausible-looking src/dst for display only (simulate mode has no real packets)."""
    if is_attack:
        src = f'185.{random.randint(1, 254)}.{random.randint(1, 254)}.{random.randint(1, 254)}'
    else:
        src = f'192.168.1.{random.randint(2, 254)}'
    dst = f'10.0.0.5:{random.choice([80, 443])}'
    return src, dst


def _make_flow_event(family_request: str, flow_id: int) -> dict:
    """Sample one real flow of the requested family, classify + explain it, build an event."""
    df = SAMPLER.sample_flows(family_request, n=1)
    scaled = STREAM_PREDICTOR.preprocess(df)
    pred = STREAM_PREDICTOR.predict(df)
    probs = pred['probabilities'][0]
    idx = int(np.argmax(probs))
    label = str(pred['labels'][0])
    reasons = EXPLAINER.explain(scaled[0], df.row(0, named=True), idx, top_k=8)
    src, dst = _synthetic_endpoints(is_attack=family_request != 'Benign')
    return {
        'flow_id': flow_id,
        'src': src,
        'dst': dst,
        'true_family': family_request,
        'family': label,
        'gate': 'allow' if label == 'Benign' else 'block',
        'confidence': float(pred['confidences'][0]),
        'probabilities': {n: float(p) for n, p in zip(pred['class_names'], probs)},
        'explanation': reasons,
    }


async def _event_stream():
    flow_id = 0
    while True:
        family = INJECT_QUEUE.popleft() if INJECT_QUEUE else 'Benign'
        flow_id += 1
        try:
            event = _make_flow_event(family, flow_id)
            yield f'data: {json.dumps(event)}\n\n'
        except Exception as e:  # never kill the stream on a single bad flow
            yield f'data: {json.dumps({"flow_id": flow_id, "error": str(e)})}\n\n'
        # Burst faster while an attack is in progress, idle slower for the benign baseline.
        await asyncio.sleep(0.15 if INJECT_QUEUE else 0.6)


class InjectRequest(BaseModel):
    family: str
    count: int = 20


def _aggregate(pred: dict) -> dict:
    """Aggregate per-flow predictions into a single result."""
    import numpy as np
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
    input_type: str        = Form('csv'),
):
    predictor_key = f'{split} / {mode}-class'
    if predictor_key not in PREDICTORS:
        available = list(PREDICTORS.keys())
        return {'error': f'Model {predictor_key!r} not found. Available: {available}'}

    t0 = time.time()
    contents = await file.read()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        if input_type == 'pcap':
            pcap_path = tmp_path / file.filename
            pcap_path.write_bytes(contents)
            try:
                csv_path = run_cicflowmeter(pcap_path, tmp_path)
            except CICFlowMeterError as e:
                return {'error': f'CICFlowMeter error: {e}'}
        else:
            csv_path = tmp_path / (file.filename or 'upload.csv')
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


# ── Live SOC demo endpoints ──────────────────────────────────────────────────
@api.get('/api/stream')
async def stream_endpoint():
    """SSE feed of classified flows for the SOC dashboard."""
    return StreamingResponse(
        _event_stream(),
        media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'},
    )


@api.post('/api/inject')
def inject_endpoint(req: InjectRequest):
    """Queue attack flows of a green family to appear in the live stream."""
    if req.family not in SAMPLER.families:
        return {'error': f'Unknown family {req.family!r}. Available: {SAMPLER.families}'}
    n = max(1, min(req.count, 200))
    for _ in range(n):
        INJECT_QUEUE.append(req.family)
    return {'queued': n, 'family': req.family, 'pending': len(INJECT_QUEUE)}


@api.get('/api/feature-importance')
def feature_importance_endpoint():
    """Global permutation feature importance for the explainability panel."""
    path = MODELS_DIR / 'perm_imp_global_test.joblib'
    if not path.exists():
        return {'error': 'perm_imp_global_test.joblib not found'}
    raw = joblib.load(path)
    features = list(STREAM_PREDICTOR.x_columns)
    if isinstance(raw, dict) and 'features' in raw and 'importance' in raw:
        names = list(raw['features'])
        imps = np.asarray(raw['importance']).reshape(-1)
        items = [{'feature': str(n), 'importance': float(i)} for n, i in zip(names, imps)]
    elif isinstance(raw, dict):
        items = [{'feature': str(k), 'importance': float(np.asarray(v).mean())}
                 for k, v in raw.items()]
    else:
        arr = np.asarray(raw).reshape(-1)
        items = [{'feature': features[i] if i < len(features) else str(i),
                  'importance': float(arr[i])} for i in range(len(arr))]
    items.sort(key=lambda d: -d['importance'])
    return {'features': items}


@api.get('/api/families')
def families_endpoint():
    return {'families': SAMPLER.families, 'classes': list(STREAM_PREDICTOR.encoder.classes_)}


# ── Gradio UI ──────────────────────────────────────────────────────────────
def _gradio_classify_csv(file, predictor_key):
    if file is None:
        return 'Upload a CSV first.', []
    pred = classify_csv(Path(file.name), predictor_key)
    rows = [[i, str(lbl), f'{c:.4f}'] for i, (lbl, c) in enumerate(zip(pred['labels'], pred['confidences']))]
    counts = {}
    for lbl in pred['labels']:
        counts[str(lbl)] = counts.get(str(lbl), 0) + 1
    summary = ' | '.join(f'{k}: {v}' for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))
    return f'{len(pred["labels"])} flows — {summary}', rows


def _gradio_classify_pcap(file, predictor_key):
    if file is None:
        return 'Upload a PCAP first.', []
    with tempfile.TemporaryDirectory() as tmp:
        try:
            csv_path = run_cicflowmeter(Path(file.name), Path(tmp))
        except CICFlowMeterError as e:
            return f'CICFlowMeter error: {e}', []
        return _gradio_classify_csv(type('F', (), {'name': str(csv_path)})(), predictor_key)


with gr.Blocks(title='RT-IDS') as demo:
    gr.Markdown('# RT-IDS — CIC-IoT-2023')
    predictor_choice = gr.Dropdown(
        choices=list(PREDICTORS.keys()),
        value=list(PREDICTORS.keys())[0],
        label='Model',
    )
    with gr.Tab('CSV upload'):
        csv_in  = gr.File(label='CIC-format CSV', file_types=['.csv'])
        csv_btn = gr.Button('Classify CSV')
        csv_summary = gr.Textbox(label='Summary', interactive=False)
        csv_table   = gr.Dataframe(headers=['#', 'Predicted', 'Confidence'], interactive=False)
        csv_btn.click(_gradio_classify_csv, [csv_in, predictor_choice], [csv_summary, csv_table])
    with gr.Tab('PCAP upload'):
        pcap_in  = gr.File(label='PCAP file', file_types=['.pcap', '.pcapng'])
        pcap_btn = gr.Button('Extract flows + classify')
        pcap_summary = gr.Textbox(label='Summary', interactive=False)
        pcap_table   = gr.Dataframe(headers=['#', 'Predicted', 'Confidence'], interactive=False)
        pcap_btn.click(_gradio_classify_pcap, [pcap_in, predictor_choice], [pcap_summary, pcap_table])

app = gr.mount_gradio_app(api, demo, path='/')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=7860)
