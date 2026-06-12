# Mock customer website (protected party). Ordinary isolated container.
FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi "uvicorn[standard]"
COPY mock_site/ ./mock_site/
EXPOSE 80
CMD ["uvicorn", "mock_site.app:app", "--host", "0.0.0.0", "--port", "80"]
