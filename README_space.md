---
title: IDS Backend
emoji: 🛡️
colorFrom: blue
colorTo: cyan
sdk: docker
pinned: false
---

# RT-IDS Backend — CIC-IoT-2023

FastAPI + Gradio inference server for real-time network intrusion detection.

- **Gradio UI**: `/`
- **Health check**: `GET /api/health`
- **Inference**: `POST /api/classify` (multipart: file, model_type, mode, split, input_type)
