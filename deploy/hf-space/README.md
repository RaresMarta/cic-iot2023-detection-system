---
title: RT-IDS Analyzer
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RT-IDS Analyzer — CIC-IoT-2023

FastAPI inference backend for the bachelor-thesis intrusion-detection system. Upload a
packet capture (`.pcap`/`.pcapng`) or a 39-feature flow CSV and the service extracts
flows, scales them the same way training did, and classifies them.

## Endpoints

- `GET  /api/health` — liveness + the list of loaded models (`model_type/split/mode`).
- `POST /api/classify` — multipart form:
  - `file` — the `.pcap`/`.pcapng`/`.csv` upload
  - `model_type` — `mlp` | `rf`   (default `mlp`)
  - `mode` — `2` | `8`            (default `2`)
  - `split` — `random`           (default `random`)

Returns the dominant label, confidence, per-class probabilities, a per-flow breakdown,
and — for the 8-class MLP — SHAP `top_features`.

## Models shipped

`mlp/random/2`, `mlp/random/8`, `rf/random/2`. The 1.3 GB `rf/random/8` forest is left
out to fit the free CPU tier; requesting it returns a clean "model not found".
