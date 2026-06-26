---
title: IDS Backend
emoji: 🛡️
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
---

# RT-IDS Backend — CIC-IoT-2023

FastAPI REST inference server for real-time network intrusion detection. The UI is a
separate React frontend (Vercel) that calls this API; this Space exposes the API only.

- **Health check**: `GET /api/health`
- **Inference**: `POST /api/classify` (multipart: file, model_type, mode, split, input_type)
