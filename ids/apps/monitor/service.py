"""FastAPI service for the live detector.

Exposes the contract the dashboard + mock site consume:
  GET /api/stream     SSE feed of {type: flow|alert|recovered} events
  GET /api/stats      counters
  GET /api/health     liveness + mode

Run: python -m ids.apps.monitor replay <pcap>   |   python -m ids.apps.monitor live --iface ids-br0
"""
from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel


from ids.runtime.predictor import MLPClassifier, RFClassifier

from . import config, producers
from .detector import Detector
from .events import Broker
from .supabase_sink import SupabaseSink


def _build_predictor(model_type: str, mode: str):
    """Build one classifier head. 'mlp' -> PyTorch net; 'rf' -> RandomForest joblib.
    Both expose the same predict() contract, so callers stay model-agnostic."""
    model_type = model_type.lower()
    if model_type == 'rf':
        return RFClassifier(config.MODELS_DIR, kind='rf', split=config.MODEL_SPLIT, mode=mode)
    if model_type == 'mlp':
        return MLPClassifier(config.MODELS_DIR, split=config.MODEL_SPLIT, mode=mode)
    raise ValueError(f"unknown model type {model_type!r}; use 'mlp' or 'rf'")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if config.DECISION_MODE == 'single':
        family_predictor = _build_predictor(config.FAMILY_MODEL, config.MODEL_MODE_FAMILY)
        gate_predictor = family_predictor
    else:
        gate_predictor = _build_predictor(config.GATE_MODEL, config.MODEL_MODE_GATE)
        family_predictor = _build_predictor(config.FAMILY_MODEL, config.MODEL_MODE_FAMILY)
    producer, inject_queue = producers.from_config()
    broker = Broker()

    detector = Detector(producer, gate_predictor, family_predictor, broker)
    app.state.detector = detector
    app.state.broker = broker
    app.state.inject_queue = inject_queue
    app.state.mode = producer.mode

    if config.DECISION_MODE == 'single':
        app.state.model = f'{config.MODEL_SPLIT} single-{config.MODEL_MODE_FAMILY}class={config.FAMILY_MODEL}'
    else:
        app.state.model = (f'{config.MODEL_SPLIT} gate={config.GATE_MODEL}-{config.MODEL_MODE_GATE}c '
                           f'family={config.FAMILY_MODEL}-{config.MODEL_MODE_FAMILY}c')
    await detector.start()
    print(f'[service] started: mode={producer.mode} model={app.state.model}', flush=True)

    app.state.supabase_task = None
    if config.SUPABASE_ENABLED and config.SUPABASE_URL and config.SUPABASE_KEY and config.MONITOR_ID:
        supa = SupabaseSink(
            config.SUPABASE_URL, config.SUPABASE_KEY, config.MONITOR_ID, config.MONITOR_NAME,
            broker, owner_id=config.MONITOR_OWNER, public_ip=config.MONITOR_PUBLIC_IP,
            protected_ips=sorted(config.PROTECTED_IPS), flow_rate=config.SUPABASE_FLOW_RATE,
            snapshot_s=config.SUPABASE_SNAPSHOT_S)
        app.state.supabase = supa
        app.state.supabase_task = asyncio.create_task(supa.run(detector))
        print(f'[service] supabase backplane enabled: monitor={config.MONITOR_ID}', flush=True)

    try:
        yield
    finally:
        if app.state.supabase_task is not None:
            app.state.supabase_task.cancel()
        await detector.stop()


app = FastAPI(title='RT-IDS Live Detector', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'],
)


@app.get('/api/health')
def health():
    return {'status': 'ok', 'mode': getattr(app.state, 'mode', '?'),
            'model': getattr(app.state, 'model', '?')}


@app.get('/api/stats')
def stats():
    return app.state.detector.snapshot_stats()


class InjectRequest(BaseModel):
    family: str
    count: int = 20


@app.post('/api/inject')
def inject(req: InjectRequest):
    """Queue attack flows of a green family into the simulate stream (simulate mode only)."""
    q = getattr(app.state, 'inject_queue', None)
    if q is None:
        return {'error': 'inject is only available in simulate mode'}
    valid = {'Benign', 'DDoS', 'DoS', 'Mirai', 'Recon'}
    if req.family not in valid:
        return {'error': f'unknown family {req.family!r}', 'valid': sorted(valid)}
    n = max(1, min(req.count, 200))
    for _ in range(n):
        q.append(req.family)
    return {'queued': n, 'family': req.family, 'pending': len(q)}


@app.get('/api/families')
def families():
    return {'families': ['Benign', 'DDoS', 'DoS', 'Mirai', 'Recon'],
            'alert': ['DDoS', 'DoS', 'Mirai', 'Recon']}


@app.get('/api/stream')
async def stream():
    broker: Broker = app.state.broker

    async def gen():
        q = broker.subscribe()
        try:
            yield ': connected\n\n'
            while True:
                try:
                    evt = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f'data: {json.dumps(evt)}\n\n'
                except asyncio.TimeoutError:
                    yield ': keepalive\n\n'
        finally:
            broker.unsubscribe(q)

    return StreamingResponse(
        gen(), media_type='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no',
                 'Connection': 'keep-alive'},
    )
