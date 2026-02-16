"""
Pydantic schemas for model-related endpoints.
"""

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Information about a loaded model."""
    
    repo_id: str = Field(..., description="Hugging Face repository ID")
    name: str = Field(..., description="Short name of the model")
    model_type: Literal["submodel", "fusion"] = Field(
        ...,
        description="Type of model"
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Model configuration from config.json"
    )


class ModelsListResponse(BaseModel):
    """Response schema for listing models."""
    
    fusion: Optional[ModelInfo] = Field(
        None,
        description="Fusion model information"
    )
    submodels: List[ModelInfo] = Field(
        default_factory=list,
        description="List of loaded submodels"
    )
    total_count: int = Field(..., description="Total number of loaded models")


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: Literal["ok", "error"] = Field(..., description="Health status")


class ReadyResponse(BaseModel):
    """Response schema for readiness check."""
    
    status: Literal["ready", "not_ready"] = Field(..., description="Readiness status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    fusion_repo: Optional[str] = Field(None, description="Fusion repository ID")
    submodels: List[str] = Field(
        default_factory=list,
        description="List of loaded submodel repository IDs"
    )
