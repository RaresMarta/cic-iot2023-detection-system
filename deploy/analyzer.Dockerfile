# Analyzer image — the stateless batch inference backend (/api/classify on :7860).
# Upload a pcap/csv, get a per-flow verdict. Separate container from the live
# detector so its SHAP/upload bursts cannot stall the real-time capture loop.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image small; the MLP is tiny. scikit-learn pinned to the
# version the RF artefacts were saved with (cross-minor unpickling can misbehave).
RUN pip install --no-cache-dir \
        "torch==2.*" --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        dpkt fastapi "uvicorn[standard]" polars pyarrow joblib numpy pandas \
        scikit-learn==1.8.0 shap python-multipart

# Project code + trained artefacts + the flow pool the SHAP explainer samples.
COPY pyproject.toml ./
COPY ids/    ./ids/
COPY models/ ./models/
COPY data/   ./data/

ENV PYTHONPATH=/app
EXPOSE 7860
CMD ["python", "-m", "ids.apps.analyzer.app"]
