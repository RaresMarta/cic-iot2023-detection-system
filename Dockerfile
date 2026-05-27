FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends default-jre-headless && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py models.py preprocessing.py labels.py ./
COPY demo/ demo/
COPY models/ models/

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["python", "-m", "demo.app"]
