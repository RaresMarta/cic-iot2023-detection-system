"""Mock customer website — the protected party (stand-in for a real client's site).

It contains NO detection code. It just serves a small page and a few endpoints for
attackers to hit. For the demo's explainability layer, the page subscribes to the
detector's SSE feed and visualises attack_detected / ip_banned / recovered — a
presentation layer only; a real on-prem deployment keeps the protected app unaware.

Run: uvicorn mock_site.app:app --host 0.0.0.0 --port 80
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

STATIC = Path(__file__).resolve().parent / 'static'

# Where the browser can reach the detector's SSE feed. The page runs in the
# visitor's browser, so this must be a host-reachable URL, not a container name.
DETECTOR_URL = os.environ.get('DETECTOR_URL', 'http://localhost:7870')
SITE_NAME = os.environ.get('SITE_NAME', 'Aperture Cloud')

app = FastAPI(title='Mock customer site')


@app.get('/', response_class=HTMLResponse)
def index():
    html = (STATIC / 'index.html').read_text()
    return html.replace('{{DETECTOR_URL}}', DETECTOR_URL).replace('{{SITE_NAME}}', SITE_NAME)


@app.get('/api/ping')
def ping():
    return {'ok': True, 'ts': time.time()}


@app.get('/api/products')
def products():
    # a little payload so GET floods have something to chew on
    return JSONResponse([{'id': i, 'name': f'Widget {i}', 'price': 9.99 + i} for i in range(20)])


@app.get('/health')
def health():
    return {'status': 'ok'}


app.mount('/static', StaticFiles(directory=str(STATIC)), name='static')
