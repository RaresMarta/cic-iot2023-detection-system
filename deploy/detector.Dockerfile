# Live detector image. Runs with network_mode: host + NET_RAW so it can sniff the
# bridge interface with real per-source IPs (passive: detect + alert, no blocking).
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small; the MLP is tiny.
RUN pip install --no-cache-dir \
        "torch==2.*" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        dpkt fastapi "uvicorn[standard]" polars joblib numpy scikit-learn

# Project code + trained artefacts. The monitor needs the whole `ids` package
# (it reuses ids.runtime / ids.data / ids.core).
COPY pyproject.toml ./
COPY ids/ ./ids/
COPY models/ ./models/

ENV PYTHONPATH=/app
EXPOSE 7870
# IDS_SOURCE=live (LiveCapture on IDS_IFACE) is set in compose for the VPS.
CMD ["python", "-m", "ids.apps.monitor", "live", "--iface", "ids-br0"]
