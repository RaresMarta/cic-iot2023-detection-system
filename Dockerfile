FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY ids/ ids/
COPY models/ models/

ENV PYTHONPATH=/app
EXPOSE 7860

CMD ["python", "-m", "ids.apps.analyzer.app"]
