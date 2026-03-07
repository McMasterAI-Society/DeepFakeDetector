# DeepFake Detector Backend

FastAPI backend for detecting AI-generated (deepfake) images.

## Features

- **Multi-model ensemble**: CNN, ViT, DeiT, and GradField models
- **Fusion prediction**: Combines submodel predictions using Logistic Regression or Meta-classifier
- **Explainability**: Grad-CAM and Attention Rollout heatmaps
- **LLM Insights**: Optional AI-powered interpretation of model evidence (Google Gemini)
- **Hugging Face integration**: Models downloaded and cached automatically

## Quick Start

### Prerequisites

- Python 3.11+
- pip

### Installation

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Locally

```bash
# From backend directory
uvicorn app.main:app --reload

# Or run directly
python -m app.main
```

The API will be available at `http://localhost:8000`

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_FUSION_REPO_ID` | `DeepFakeDetector/fusion-logreg-final` | Hugging Face fusion model repo |
| `HF_CACHE_DIR` | `.hf_cache` | Local cache directory for HF models |
| `HF_TOKEN` | `None` | HF API token (for private repos) |
| `GOOGLE_API_KEY` | `None` | Google Gemini API key (for LLM explanations) |
| `CORS_ORIGINS` | `http://localhost:5173,...` | Comma-separated allowed CORS origins |
| `ENABLE_DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

Available fusion models:
- `DeepFakeDetector/fusion-logreg` - Logistic Regression (default)
- `DeepFakeDetector/fusion-meta-classifier` - Neural network meta-classifier

Create a `.env` file in the backend directory to set these:

```env
HF_FUSION_REPO_ID=DeepFakeDetector/fusion-logreg-final
HF_CACHE_DIR=.hf_cache
CORS_ORIGINS=http://localhost:5173,https://www.deepfake-detector.app
ENABLE_DEBUG=true
LOG_LEVEL=DEBUG
```

## API Endpoints

### Health & Status

```bash
# Health check
curl http://localhost:8000/health

# Readiness check (verifies models are loaded)
curl http://localhost:8000/ready
```

### List Models

```bash
curl http://localhost:8000/models
```

### Predict

```bash
# Using fusion (default) - combines all submodel predictions
curl -X POST http://localhost:8000/predict \
  -F "image=@/path/to/image.jpg"

# Using fusion without submodel details in response
curl -X POST "http://localhost:8000/predict?return_submodels=false" \
  -F "image=@/path/to/image.jpg"

# Using a specific submodel (no fusion)
curl -X POST "http://localhost:8000/predict?use_fusion=false&model=test-random-a" \
  -F "image=@/path/to/image.jpg"
```

### Example Response

```json
{
  "final": {
    "pred": "fake",
    "pred_int": 1,
    "prob_fake": 0.6667
  },
  "fusion_used": true,
  "submodels": {
    "test-random-a": {"pred": "real", "pred_int": 0, "prob_fake": 0.0},
    "test-random-b": {"pred": "fake", "pred_int": 1, "prob_fake": 1.0},
    "test-random-c": {"pred": "fake", "pred_int": 1, "prob_fake": 1.0}
  },
  "timing_ms": {
    "total": 7,
    "download": 1,
    "preprocess": 0,
    "inference": 2,
    "fusion": 0
  }
}
```

## Docker

### Build

```bash
docker build -t deepfake-detector-api .
```

### Run

```bash
docker run -p 8000:8000 deepfake-detector-api
```

### With Environment Variables

```bash
docker run -p 8000:8000 \
  -e HF_FUSION_REPO_ID=DeepFakeDetector/fusion-logreg \
  -e LOG_LEVEL=DEBUG \
  deepfake-detector-api
```

### With Volume for Cache Persistence

```bash
docker run -p 8000:8000 \
  -v $(pwd)/.hf_cache:/app/.hf_cache \
  deepfake-detector-api
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api.py -v
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── api/                  # API route handlers
│   │   ├── routes_health.py  # /health, /ready endpoints
│   │   ├── routes_models.py  # /models endpoint
│   │   └── routes_predict.py # /predict endpoint
│   ├── core/                 # Core configuration
│   │   ├── config.py         # Settings and env vars
│   │   ├── errors.py         # Custom exceptions
│   │   └── logging.py        # Logging setup
│   ├── schemas/              # Pydantic models
│   │   ├── predict.py        # Prediction schemas
│   │   └── models.py         # Model info schemas
│   ├── services/             # Business logic
│   │   ├── hf_hub_service.py # HuggingFace Hub downloads
│   │   ├── model_registry.py # Model loading & management
│   │   ├── preprocess_service.py # Image preprocessing
│   │   ├── inference_service.py  # Model inference
│   │   ├── fusion_service.py     # Fusion predictions
│   │   └── cache_service.py      # Caching (placeholder)
│   ├── models/
│   │   └── wrappers/         # Model wrapper classes
│   │       ├── base_wrapper.py
│   │       ├── dummy_random_wrapper.py
│   │       └── dummy_majority_fusion_wrapper.py
│   └── utils/                # Utilities
│       ├── image.py          # Image processing
│       ├── timing.py         # Performance timing
│       └── security.py       # Security utilities
├── tests/                    # Test suite
├── Dockerfile
├── requirements.txt
└── README.md
```

## Hugging Face Model Repositories

### Fusion Models
- `DeepFakeDetector/fusion-logreg-final` - Logistic Regression (default)
- `DeepFakeDetector/fusion-meta-final` - Neural network meta-classifier
- Each contains: `config.json`, `predict.py`
- Function: Combines submodel predictions into final verdict

### Submodels
- `DeepFakeDetector/cnn-transfer-final` - EfficientNet-B0 CNN
- `DeepFakeDetector/vit-base-final` - Vision Transformer
- `DeepFakeDetector/deit-distilled-final` - Data-efficient Image Transformer
- `DeepFakeDetector/gradfield-cnn-final` - Gradient field analysis CNN
- Each contains: `config.json`, `model.pt`, `predict.py`

## Future Milestones

- **Milestone 2**: Real CNN/ViT models for deepfake detection ✓
- **Milestone 3**: Explainability endpoints ✓
- **Milestone 4**: Production optimizations

## License

MIT
