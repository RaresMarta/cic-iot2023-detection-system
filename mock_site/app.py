"""Mock customer website — the protected party (stand-in for a real client's site).

It contains NO detection code and is unaware of the detector: a real on-prem
protected app does not subscribe to its own monitor. It just serves a few plain
HTTP endpoints for attackers to hit so the passive sniffer has traffic to classify.

Run: uvicorn mock_site.app:app --host 0.0.0.0 --port 80
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse

app = FastAPI(title='Mock customer site')


@app.get('/', response_class=PlainTextResponse)
def index():
    return 'Aperture Cloud — OK'


@app.get('/api/ping')
def ping():
    return {'ok': True, 'ts': time.time()}


@app.get('/api/products')
def products():
    return JSONResponse([{'id': i, 'name': f'Widget {i}', 'price': 9.99 + i} for i in range(20)])


@app.get('/health')
def health():
    return {'status': 'ok'}
