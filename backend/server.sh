#!/bin/bash
# Production server startup script for Railway/Render
# This script starts the FastAPI backend with Uvicorn

# Get PORT from environment variable (set by Railway/Render)
# Default to 8000 if not set
PORT=${PORT:-8000}

# Limit PyTorch / OpenMP threads to prevent massive memory allocations
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Start Uvicorn with production settings
python -m uvicorn backend.api.server:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 1 \
  --loop uvloop \
  --http httptools
