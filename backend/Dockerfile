FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ backend/api/
COPY src/ backend/src/
COPY __init__.py backend/__init__.py

EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.api.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
