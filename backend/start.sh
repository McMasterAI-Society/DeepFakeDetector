#!/bin/bash
# Startup script for deployment
# Loads .env file if present, then uses PORT environment variable
# Defaults: 7860 (HuggingFace Spaces) or 8000

# Load .env file if it exists
if [ -f .env ]; then
    echo "Loading environment from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Use PORT from env, .env file, or default to 7860
PORT=${PORT:-7860}

echo "Starting uvicorn on port $PORT"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --log-level info
