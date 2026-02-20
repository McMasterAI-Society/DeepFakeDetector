"""
Pydantic schemas for prediction endpoints.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field


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
