"""
Pydantic schemas for model-related endpoints.
"""

from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field


class ModelServingInfo(BaseModel):
    """Which artifact of a model is being served."""

    repo_id: str = Field(..., description="Hugging Face repository ID")
    name: str = Field(..., description="Short name of the model")
    model_type: Literal["submodel", "fusion", "probe", "backbone"] = Field(
        ...,
        description="Type of model: fusion/submodel (legacy-fusion mode) or probe/backbone (foundation-probe mode)"
    )
    revision: Optional[str] = Field(
        None,
        description="Resolved commit sha if known, otherwise the requested revision"
    )
    requested_revision: Optional[str] = Field(
        None,
        description="Revision requested by config/manifest (None means main)"
    )
    weights_file: Optional[str] = Field(None, description="Loaded weights file")
    input_size: Optional[int] = Field(None, description="Square input size in pixels")


class ModelInfo(ModelServingInfo):
    """Information about a loaded model."""

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
    manifest_hash: Optional[str] = Field(
        None,
        description="Hash of served repo ids, revisions, weights files and input sizes"
    )


class HealthResponse(BaseModel):
    """Response schema for health check."""
    
    status: Literal["ok", "error"] = Field(..., description="Health status")
    manifest_hash: Optional[str] = Field(
        None,
        description="Hash of the served model manifest (None until models are loaded)"
    )
    models: List[ModelServingInfo] = Field(
        default_factory=list,
        description="Served repo, revision, weights file and input size per model"
    )


class ReadyResponse(BaseModel):
    """Response schema for readiness check."""
    
    status: Literal["ready", "not_ready"] = Field(..., description="Readiness status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    fusion_repo: Optional[str] = Field(None, description="Fusion repository ID")
    submodels: List[str] = Field(
        default_factory=list,
        description="List of loaded submodel repository IDs"
    )
