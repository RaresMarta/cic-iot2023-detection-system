FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py models.py preprocessing.py labels.py ./
COPY demo/ demo/
COPY models/ models/

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["python", "-m", "demo.app"]
