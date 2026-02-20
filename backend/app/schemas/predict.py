"""
Pydantic schemas for prediction endpoints.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# LLM Explanation schemas
class ModelInsight(BaseModel):
    """LLM-generated insight for a single model's prediction."""
    
    what_model_relied_on: str = Field(
        ...,
        description="One sentence describing what the model focused on"
    )
    possible_cues: List[str] = Field(
        ...,
        description="2-4 possible visual cues to check (with hedging language)"
    )
    confidence_note: str = Field(
        ...,
        description="Note about confidence based on prob_fake and focus pattern"
    )


class ExplanationResult(BaseModel):
    """LLM-generated explanation for all model predictions."""
    
    per_model_insights: Dict[str, ModelInsight] = Field(
        ...,
        description="Insights keyed by model name"
    )
    consensus_summary: List[str] = Field(
        ...,
        description="2-3 bullets summarizing where models agreed/disagreed"
    )


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
    timing_ms: TimingInfo = Field(
        ...,
        description="Timing breakdown in milliseconds"
    )
    explanation: Optional[ExplanationResult] = Field(
        None,
        description="LLM-generated explanation of model predictions (when generate_explanation=true)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "final": {"pred": "fake", "pred_int": 1, "prob_fake": 0.6667},
                "fusion_used": True,
                "submodels": {
                    "test-random-a": {"pred": "real", "pred_int": 0, "prob_fake": 0.0},
                    "test-random-b": {"pred": "fake", "pred_int": 1, "prob_fake": 1.0},
                    "test-random-c": {"pred": "fake", "pred_int": 1, "prob_fake": 1.0}
                },
                "timing_ms": {"total": 7, "inference": 2, "fusion": 0}
            }
        }


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict] = Field(None, description="Additional error details")
