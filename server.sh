#!/bin/bash
# Production server startup script for Railway/Render
# This script starts the FastAPI backend with Uvicorn

# Get PORT from environment variable (set by Railway/Render)
# Default to 8000 if not set
PORT=${PORT:-8000}

# Start Uvicorn with production settings
python -m uvicorn api.server:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 4 \
  --loop uvloop \
  --http httptools
