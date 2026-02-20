"""
Model listing routes.
"""

from fastapi import APIRouter

from app.core.logging import get_logger
from app.schemas.models import ModelsListResponse, ModelInfo
from app.services.model_registry import get_model_registry

logger = get_logger(__name__)
router = APIRouter(tags=["models"])


@router.get(
    "/models",
    response_model=ModelsListResponse,
    summary="List loaded models",
    description="Get information about all loaded models including fusion and submodels"
)
async def list_models() -> ModelsListResponse:
    """
    List all loaded models.
    
    Returns information about the fusion model and all submodels,
    including their Hugging Face repository IDs and configurations.
    """
    registry = get_model_registry()
    models = registry.list_models()
    
    fusion_info = None
    submodels_info = []
    
    for model in models:
        model_info = ModelInfo(
            repo_id=model["repo_id"],
            name=model["name"],
            model_type=model["model_type"],
            config=model.get("config")
        )
        
        if model["model_type"] == "fusion":
            fusion_info = model_info
        else:
            submodels_info.append(model_info)
    
    return ModelsListResponse(
        fusion=fusion_info,
        submodels=submodels_info,
        total_count=len(models)
    )
