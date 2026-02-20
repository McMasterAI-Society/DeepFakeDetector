# DeepFake Detector Backend

FastAPI backend for detecting AI-generated (deepfake) images.

## Milestone 1: Hugging Face Hosted Dummy Models

This initial milestone implements the API infrastructure using dummy random models hosted on Hugging Face for testing purposes.

### Features

- **Fusion prediction**: Combines multiple model predictions using majority vote
- **Individual model prediction**: Run specific submodels directly  
- **Timing information**: Detailed performance metrics for each request
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
| `HF_FUSION_REPO_ID` | `DeepFakeDetector/fusion-majority-test` | Hugging Face fusion model repo |
| `HF_CACHE_DIR` | `.hf_cache` | Local cache directory for HF models |
| `HF_TOKEN` | `None` | HF API token (for private repos) |
| `ENABLE_DEBUG` | `false` | Enable debug mode |
| `LOG_LEVEL` | `INFO` | Logging level |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

Create a `.env` file in the backend directory to set these:

```env
HF_FUSION_REPO_ID=DeepFakeDetector/fusion-majority-test
HF_CACHE_DIR=.hf_cache
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
  -e HF_FUSION_REPO_ID=DeepFakeDetector/fusion-majority-test \
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

### Fusion Model
- Repository: `DeepFakeDetector/fusion-majority-test`
- Contains: `config.json`, `fusion.py`
- Function: Majority vote across submodels

### Submodels
- `DeepFakeDetector/test-random-a`
- `DeepFakeDetector/test-random-b`  
- `DeepFakeDetector/test-random-c`
- Each contains: `config.json`, `predict.py`
- Function: Random 0/1 prediction (for testing)

## Future Milestones

- **Milestone 2**: Real CNN/ViT models for deepfake detection
- **Milestone 3**: Explainability endpoints
- **Milestone 4**: Production optimizations

## License

MIT
