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

# Ensure OpenMP thread count is a valid integer to avoid libgomp warnings.
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[0-9]+$ ]]; then
    export OMP_NUM_THREADS=1
fi

# Prefer persistent HF cache on Spaces when /data is available and writable.
PERSIST_ROOT="${HF_PERSISTENT_CACHE_ROOT:-/data}"
if [ "${HF_PERSISTENT_CACHE:-true}" = "true" ] && [ -d "$PERSIST_ROOT" ] && [ -w "$PERSIST_ROOT" ]; then
    if [ -z "${HF_HOME:-}" ] || [ "${HF_HOME}" = "/app/.cache/huggingface" ] || [ "${HF_HOME}" = "/app/.hf_cache/hf_home" ] || [ "${HF_HOME}" = ".cache/huggingface" ]; then
        export HF_HOME="$PERSIST_ROOT/.cache/huggingface"
    fi
    if [ -z "${HF_CACHE_DIR:-}" ] || [ "${HF_CACHE_DIR}" = ".hf_cache" ] || [ "${HF_CACHE_DIR}" = "/app/.hf_cache" ]; then
        export HF_CACHE_DIR="$PERSIST_ROOT/.hf_cache"
    fi
else
    export HF_HOME="${HF_HOME:-/app/.hf_cache/hf_home}"
    export HF_CACHE_DIR="${HF_CACHE_DIR:-/app/.hf_cache}"
fi

mkdir -p "$HF_HOME" "$HF_CACHE_DIR" 2>/dev/null || true
echo "Using HF cache dir: $HF_CACHE_DIR"
echo "Using HF home dir: $HF_HOME"

echo "Starting uvicorn launcher on port $PORT"
exec python -m app.scripts.run_server
