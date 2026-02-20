"""
Health check routes.
"""

from fastapi import APIRouter

from app.core.logging import get_logger
from app.schemas.models import HealthResponse, ReadyResponse
from app.services.model_registry import get_model_registry

logger = get_logger(__name__)
router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Simple health check to verify the API is running"
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns OK if the API server is running.
    """
    return HealthResponse(status="ok")


@router.get(
    "/ready",
    response_model=ReadyResponse,
    summary="Readiness check",
    description="Check if models are loaded and the API is ready to serve predictions"
)
async def readiness_check() -> ReadyResponse:
    """
    Readiness check endpoint.
    
    Verifies that models are loaded and ready for inference.
    Returns detailed information about loaded models.
    """
    registry = get_model_registry()
    
    if not registry.is_loaded:
        return ReadyResponse(
            status="not_ready",
            models_loaded=False,
            fusion_repo=None,
            submodels=[]
        )
    
    return ReadyResponse(
        status="ready",
        models_loaded=True,
        fusion_repo=registry.get_fusion_repo_id(),
        submodels=[
            model["repo_id"] 
            for model in registry.list_models() 
            if model["model_type"] == "submodel"
        ]
    )
