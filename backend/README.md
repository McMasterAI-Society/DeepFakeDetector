---
title: DeepFake Detector API
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DeepFake Detector Backend

FastAPI backend for detecting AI-generated (deepfake) images using a multi-model fusion pipeline.

## Features

- Multi-model ensemble: CNN Transfer, ViT Base, DeiT Distilled, Gradient Field CNN
- Fusion prediction: Logistic Regression and meta-classifier variants
- Explainability: Grad-CAM and attention-based heatmaps
- Optional Gemini-powered interpretation layer
- Hugging Face Hub model download/caching

## Prerequisites

- Python 3.11+
- pip

## Quick Start (Local)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows PowerShell: .\\venv\\Scripts\\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure env
cp .env.example .env  # Windows PowerShell: Copy-Item .env.example .env

# Run API
uvicorn app.main:app --reload
```

API docs:

- `http://localhost:8000/docs`
- `http://localhost:8000/redoc`

## Environment Configuration

Use [backend/.env.example](.env.example) as the source of truth.

Common runtime variables:

- `HF_FUSION_REPO_ID` (default: `DeepFakeDetector/fusion-logreg-final`)
- `HF_CACHE_DIR` (default: `/app/.hf_cache`, auto-switched to `/data/.hf_cache` on Spaces when mounted)
- `HF_HOME` (default: `/app/.cache/huggingface`, auto-switched to `/data/.cache/huggingface` on Spaces)
- `HF_PERSISTENT_CACHE` (default: `true`)
- `HF_PERSISTENT_CACHE_ROOT` (default: `/data`)
- `HF_TOKEN` (optional; required for private model repos or non-interactive HF auth)
- `GOOGLE_API_KEY` (optional; required for Gemini explanations)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000` locally, `7860` in Space by default)
- `CORS_ORIGINS` (comma-separated origins)
- `ENABLE_DEBUG`, `LOG_LEVEL`

HF Spaces deploy variables (used by [backend/deploy-to-hf.sh](deploy-to-hf.sh)):

- `HF_SPACE_URL`
- `HF_SPACE_WEB_URL`
- `HF_SPACE_APP_URL`
- `HF_BASE_IMAGE` (default: `python:3.11-slim`; set to your prebuilt dependency image for faster deploys)
- `HF_DEPLOY_DIR`
- `INSTALL_OPTIONAL_DEPS` (`true`/`false`, default `false`)
- `HF_BUCKET_SYNC_ON_DEPLOY` (default: `true`)

## API Endpoints

- `GET /health` - service health
- `GET /ready` - readiness (includes model load state)
- `GET /models` - loaded model metadata
- `POST /predict` - real/fake prediction
- `GET /docs` - Swagger UI

Example:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "image=@/path/to/image.jpg"
```

## Docker

Build and run locally:

```bash
docker build -t deepfake-detector-api .
docker run -p 7860:7860 deepfake-detector-api
```

Fast deploy workflow (recommended):

1. Build and push dependency base image when dependencies change:

```bash
docker build -f Dockerfile.base -t ghcr.io/<org>/<repo>-backend-base:latest .
docker push ghcr.io/<org>/<repo>-backend-base:latest
```

2. Set this in `backend/.env` before deploy:

```bash
HF_BASE_IMAGE=ghcr.io/<org>/<repo>-backend-base:latest
```

3. Deploy normally:

```bash
bash ./backend/deploy-to-hf.sh
```

Image defaults:

- Installs `requirements-runtime.txt` (inference + explainability + LLM)
- Skips build-time model prefetch (first boot fills `/data` cache)
- `POST /predict` defaults to `explain=true`

Enable optional test dependencies at build time:

```bash
docker build --build-arg INSTALL_OPTIONAL_DEPS=true -t deepfake-detector-api .
```

## Deploy to Hugging Face Spaces

Recommended path is the Bash deploy script.

1. Configure [backend/.env](.env) from [backend/.env.example](.env.example)
2. Ensure `HF_SPACE_URL` and related deploy variables are set
3. Run from the repo root:

```bash
bash ./backend/deploy-to-hf.sh
```

The script will:

- sync `backend/data` to your HF bucket first (when configured)
- install Hugging Face CLI if needed
- prompt/authenticate with HF (`hf auth login`) when required
- clone Space repo into a separate temp deploy directory
- copy backend files as-is (single `Dockerfile` setup)
- commit and push to the HF Space

If bucket sync should be skipped for a deploy, set:

- `HF_BUCKET_SYNC_ON_DEPLOY=false`

After deploy, set Space secrets in Hugging Face:

- `GOOGLE_API_KEY` (if using explanation endpoints)
- `CORS_ORIGINS` (frontend domains)

## Persistent Bucket Cache on Spaces

The backend now prefers persistent cache paths when a bucket is mounted read/write at `/data`.

Recommended bucket mount settings in Space:

- Mount path: `/data`
- Access mode: `Read & Write`

At runtime, `start.sh` will auto-select:

- `HF_HOME=/data/.cache/huggingface`
- `HF_CACHE_DIR=/data/.hf_cache`

If `/data` is unavailable, it falls back to `/app/.cache/huggingface` and `/app/.hf_cache`.

## Bucket Upload / Sync (Reproducible)

Set these optional values in `backend/.env`:

- `HF_BUCKET_URI=hf://buckets/lukhsaankumar/DeepFakeDetectorBackend-storage`
- `HF_BUCKET_LOCAL_DIR=./data`
- `HF_BUCKET_DELETE=false`
- `HF_BUCKET_SYNC_ON_DEPLOY=true`

Bash:

```bash
cd backend
chmod +x ./sync-bucket.sh
./sync-bucket.sh              # uses HF_BUCKET_LOCAL_DIR or ./data
./sync-bucket.sh ./data       # explicit local path
```

PowerShell:

```powershell
cd backend
./sync-bucket.ps1             # uses HF_BUCKET_LOCAL_DIR or ./data
./sync-bucket.ps1 .\data      # explicit local path
```

Notes:

- Scripts require authenticated HF CLI (`hf auth login`).
- Set `HF_BUCKET_DELETE=true` to mirror local to remote (deletes remote files not present locally).

## Deploy to Railway

- Set service root to `backend`
- Configure required env vars (`CORS_ORIGINS`, `HF_FUSION_REPO_ID`, `HF_CACHE_DIR`, `PORT`)
- Push to main branch to trigger deployment

## Troubleshooting

- Build fails: inspect Space/Railway logs first
- HF auth fails: run `hf auth login` and/or set `HF_TOKEN` in `.env`
- Model loading issues: verify fusion/submodel repo IDs and access
- CORS issues: ensure frontend domains are in `CORS_ORIGINS`

## Project Structure

```text
backend/
├── app/
├── data/
├── tests/
├── Dockerfile
├── deploy-to-hf.sh
├── sync-bucket.sh
├── sync-bucket.ps1
├── requirements.txt
└── README.md
```

## License

MIT
