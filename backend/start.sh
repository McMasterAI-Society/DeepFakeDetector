#!/bin/bash
# Startup script for Railway deployment
# Uses PORT environment variable if set, otherwise defaults to 8000

PORT=${PORT:-8000}

echo "Starting uvicorn on port $PORT"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --log-level info
