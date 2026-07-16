FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ backend/api/
COPY src/ backend/src/
COPY __init__.py backend/__init__.py
COPY server.sh backend/server.sh

RUN chmod +x backend/server.sh

EXPOSE 8000

CMD ["bash", "backend/server.sh"]
