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


from ids.runtime.predictor import MLPClassifier  # noqa: E402

from . import config, producers  # noqa: E402
from .detector import Detector  # noqa: E402
from .events import Broker  # noqa: E402
from .store import SqliteSink  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    gate_predictor = MLPClassifier(config.MODELS_DIR, split=config.MODEL_SPLIT, mode=config.MODEL_MODE_GATE)
    family_predictor = MLPClassifier(config.MODELS_DIR, split=config.MODEL_SPLIT, mode=config.MODEL_MODE_FAMILY)
    producer, inject_queue = producers.from_config()
    broker = Broker()

    # Per-alert SHAP on the gate verdict; degrades to the saliency proxy if shap or
    # the background parquet is unavailable (e.g. a slim live image without the data).
    explainer = None
    try:
        from ids.runtime.explain import FlowExplainer
        from ids.data.sampler import FlowSampler
        explainer = FlowExplainer(gate_predictor, FlowSampler())
        print('[service] live SHAP enabled (gate explainer)', flush=True)
    except Exception as e:
        print(f'[service] live SHAP unavailable, alerts use saliency proxy: {e}', flush=True)

    detector = Detector(producer, gate_predictor, family_predictor, broker, explainer)
    app.state.detector = detector
    app.state.broker = broker
    app.state.inject_queue = inject_queue          # None unless simulate mode
    app.state.mode = producer.mode
    app.state.model = f'{config.MODEL_SPLIT} gate={config.MODEL_MODE_GATE} family={config.MODEL_MODE_FAMILY}'
    await detector.start()
    print(f'[service] started: mode={producer.mode} model={app.state.model}', flush=True)

    # Optional event store: a second broker consumer that persists incidents + stats.
    # Default-off; never on the detection path (see store.py).
    app.state.store = None
    app.state.store_task = None
    if config.DB_ENABLED:
        sink = SqliteSink(config.DB_PATH, broker, snapshot_s=config.DB_SNAPSHOT_S)
        app.state.store = sink
        app.state.store_task = asyncio.create_task(sink.run(detector))
        print(f'[service] event store enabled: {config.DB_PATH}', flush=True)

    try:
        yield
    finally:
        if app.state.store_task is not None:
            app.state.store_task.cancel()
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


@app.get('/api/incidents')
def incidents(limit: int = 50):
    """Recent attack episodes from the event store (empty if the store is disabled)."""
    store = getattr(app.state, 'store', None)
    if store is None:
        return {'enabled': False, 'incidents': []}
    return {'enabled': True, 'incidents': store.recent_incidents(limit)}


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
            # initial comment keeps some proxies from buffering the stream open
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
