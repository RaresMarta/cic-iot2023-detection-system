# Hugging Face Space — RT-IDS analyzer backend (CIC-IoT-2023).
# Docker SDK Space: HF builds this image and routes traffic to app_port (7860).
# This is the BATCH analyzer (upload pcap/csv -> classify). The live :7870
# detector is NOT here — it needs raw host capture and runs on a Linux VPS.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small; the MLP is tiny. Installed separately so
# the heavy wheel comes from PyTorch's CPU index, not PyPI.
# On the CPU index, 2.5.1 IS the CPU wheel (the +cpu suffix only exists from 2.6.0 on).
RUN pip install --no-cache-dir \
        torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Project code + trained artefacts + the flow pool the SHAP explainer samples.
COPY pyproject.toml .
COPY ids/    ./ids/
COPY models/ ./models/
COPY data/   ./data/

ENV PYTHONPATH=/app
# HF routes traffic to app_port (see README.md frontmatter); the analyzer binds 7860.
EXPOSE 7860
CMD ["python", "-m", "ids.apps.analyzer.app"]
