"""
FastAPI application entry point.

DeepFake Detector API - Milestone 1: Hugging Face hosted dummy models.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_health, routes_models, routes_predict
from app.core.config import settings
from app.core.errors import DeepFakeDetectorError
from app.core.logging import setup_logging, get_logger
from app.services.model_registry import get_model_registry

# Set up logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events:
    - Startup: Load models from Hugging Face
    - Shutdown: Cleanup resources
    """
    # Startup
    logger.info("Starting DeepFake Detector API...")
    logger.info(f"Configuration: HF_FUSION_REPO_ID={settings.HF_FUSION_REPO_ID}")
    logger.info(f"Configuration: HF_CACHE_DIR={settings.HF_CACHE_DIR}")
    
    # Load models from Hugging Face
    try:
        registry = get_model_registry()
        await registry.load_from_fusion_repo(settings.HF_FUSION_REPO_ID)
        logger.info("Models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load models on startup: {e}")
        logger.warning("API will start but /ready will report not_ready until models are loaded")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down DeepFake Detector API...")


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    DeepFake Detector API - Analyze images to detect AI-generated content.
    
    ## Features
    
    - **Fusion prediction**: Combines multiple model predictions using majority vote
    - **Individual model prediction**: Run specific submodels directly
    - **Timing information**: Detailed performance metrics for each request
    
    ## Milestone 1
    
    This is the initial milestone using dummy random models hosted on Hugging Face
    for testing the API infrastructure.
    """,
    lifespan=lifespan,
    debug=settings.ENABLE_DEBUG
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler for custom errors
@app.exception_handler(DeepFakeDetectorError)
async def deepfake_error_handler(request: Request, exc: DeepFakeDetectorError):
    """Handle custom DeepFakeDetector exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": exc.message,
            "details": exc.details
        }
    )


# Include routers
app.include_router(routes_health.router)
app.include_router(routes_models.router)
app.include_router(routes_predict.router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.ENABLE_DEBUG
    )
