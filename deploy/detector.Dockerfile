# Live detector image. Runs with network_mode: host + NET_ADMIN/NET_RAW so it can
# sniff the bridge interface and write nftables on the host.
FROM python:3.11-slim

# nftables for enforcement; libpcap not needed (raw AF_PACKET socket via stdlib).
RUN apt-get update && apt-get install -y --no-install-recommends \
        nftables \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image small; the MLP is tiny.
RUN pip install --no-cache-dir \
        "torch==2.*" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        dpkt fastapi "uvicorn[standard]" polars joblib numpy scikit-learn

# Project code + trained artefacts.
COPY config.py labels.py models.py ./
COPY demo/ ./demo/
COPY live_detector/ ./live_detector/
COPY models/ ./models/

EXPOSE 7870
# IDS_SOURCE=live (LiveCapture on IDS_IFACE) is set in compose for the VPS.
CMD ["python", "-m", "live_detector", "live", "--iface", "ids-br0"]
