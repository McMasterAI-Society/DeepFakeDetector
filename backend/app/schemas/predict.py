"""
Pydantic schemas for prediction endpoints.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# Model display info schema (for frontend)
class ModelDisplayInfo(BaseModel):
    """User-friendly display information for a model."""
    
    display_name: str = Field(..., description="User-friendly model name (e.g., 'Texture Analysis')")
    short_name: str = Field(..., description="Short identifier (e.g., 'CNN')")
    method_name: str = Field(..., description="Explainability method name (e.g., 'Grad-CAM')")
    method_description: str = Field(..., description="Brief description of the method")
    educational_text: str = Field(..., description="Educational text about what this model analyzes")
    what_it_looks_for: List[str] = Field(..., description="List of things this model looks for")


# LLM single-model explanation schema (for on-demand requests)
class SingleModelInsight(BaseModel):
    """LLM-generated insight for a single model (on-demand)."""
    
    key_finding: str = Field(..., description="Main finding from the model")
    what_model_saw: str = Field(..., description="What the model detected in the image")
    important_regions: List[str] = Field(..., description="Key regions identified")
    confidence_qualifier: str = Field(..., description="Confidence assessment with hedging")


class PredictionResult(BaseModel):
    """Single prediction result from a model."""
    
    pred: Literal["real", "fake"] = Field(
        ...,
        description="Human-readable prediction label"
    )
    pred_int: Literal[0, 1] = Field(
        ...,
        description="Integer prediction: 0=real, 1=fake"
    )
    prob_fake: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the image is fake (0.0-1.0)"
    )
    heatmap_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG heatmap showing model attention/saliency (when explain=true)"
    )
    explainability_type: Optional[Literal["grad_cam", "attention_rollout"]] = Field(
        None,
        description="Type of explainability method used"
    )
    focus_summary: Optional[str] = Field(
        None,
        description="Brief description of where the model focused (e.g., 'concentrated on face region')"
    )
    contribution_percentage: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="How much this model contributed to the fusion decision (0-100%)"
    )


class FusionMeta(BaseModel):
    """Metadata from fusion model about how decision was made."""
    
    submodel_weights: Dict[str, float] = Field(
        default_factory=dict,
        description="Learned coefficients for each submodel"
    )
    weighted_contributions: Dict[str, float] = Field(
        default_factory=dict,
        description="Actual contribution to this prediction (weight * prob_fake)"
    )
    contribution_percentages: Dict[str, float] = Field(
        default_factory=dict,
        description="Normalized percentages for display"
    )


class TimingInfo(BaseModel):
    """Timing breakdown for the prediction request."""
    
    total: int = Field(..., description="Total time in milliseconds")
    download: Optional[int] = Field(None, description="Image download time in ms")
    preprocess: Optional[int] = Field(None, description="Preprocessing time in ms")
    inference: Optional[int] = Field(None, description="Model inference time in ms")
    fusion: Optional[int] = Field(None, description="Fusion computation time in ms")


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    
    final: PredictionResult = Field(
        ...,
        description="Final prediction result"
    )
    fusion_used: bool = Field(
        ...,
        description="Whether fusion was used for this prediction"
    )
    submodels: Optional[Dict[str, PredictionResult]] = Field(
        None,
        description="Individual submodel predictions (when fusion_used=true and return_submodels=true)"
    )
    fusion_meta: Optional[FusionMeta] = Field(
        None,
        description="Fusion metadata including model weights and contributions"
    )
    model_display_info: Optional[Dict[str, ModelDisplayInfo]] = Field(
        None,
        description="Display information for each model (for frontend rendering)"
    )
    timing_ms: TimingInfo = Field(
        ...,
        description="Timing breakdown in milliseconds"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "final": {"pred": "fake", "pred_int": 1, "prob_fake": 0.6667},
                "fusion_used": True,
                "submodels": {
                    "cnn-transfer": {"pred": "fake", "pred_int": 1, "prob_fake": 0.82, "contribution_percentage": 45.2},
                    "vit-base": {"pred": "fake", "pred_int": 1, "prob_fake": 0.75, "contribution_percentage": 32.1},
                    "gradfield-cnn": {"pred": "fake", "pred_int": 1, "prob_fake": 0.91, "contribution_percentage": 22.7}
                },
                "timing_ms": {"total": 250, "inference": 200, "fusion": 5}
            }
        }


# Request/Response for single-model explanation endpoint
class ExplainModelRequest(BaseModel):
    """Request schema for single-model explanation."""
    
    model_name: str = Field(..., description="Name of the model to explain")
    prob_fake: float = Field(..., ge=0.0, le=1.0, description="Model's fake probability")
    heatmap_base64: Optional[str] = Field(None, description="Base64-encoded heatmap")
    focus_summary: Optional[str] = Field(None, description="Where the model focused")
    contribution_percentage: Optional[float] = Field(None, description="Model's contribution to fusion")


class ExplainModelResponse(BaseModel):
    """Response schema for single-model explanation."""
    
    model_name: str = Field(..., description="Internal model name")
    insight: SingleModelInsight = Field(..., description="LLM-generated insight")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
