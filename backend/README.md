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
- `HF_CACHE_DIR` (default: `.hf_cache`)
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
- `HF_DEPLOY_DIR`

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

## Deploy to Hugging Face Spaces

Recommended path is the Bash deploy script.

1. Configure [backend/.env](.env) from [backend/.env.example](.env.example)
2. Ensure `HF_SPACE_URL` and related deploy variables are set
3. Run from backend folder:

```bash
bash ./deploy-to-hf.sh
```

Or run from repo root:

```bash
bash ./backend/deploy-to-hf.sh
```

The script will:

- install Hugging Face CLI if needed
- prompt/authenticate with HF (`hf auth login`) when required
- clone Space repo into a separate temp deploy directory
- copy backend files as-is (single `Dockerfile` setup)
- commit and push to the HF Space

After deploy, set Space secrets in Hugging Face:

- `GOOGLE_API_KEY` (if using explanation endpoints)
- `CORS_ORIGINS` (frontend domains)

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
├── tests/
├── Dockerfile
├── deploy-to-hf.sh
├── deploy-to-hf.ps1
├── requirements.txt
└── README.md
```

## License

MIT
├── deploy-to-hf.sh
├── deploy-to-hf.ps1
├── requirements.txt
└── README.md
```

## License

MIT
