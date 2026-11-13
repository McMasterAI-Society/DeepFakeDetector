"""
Pydantic schemas for user feedback and admin endpoints.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FeedbackRequest(BaseModel):
    """User answer to "Do you know whether this image is AI-generated?"."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(..., min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    user_knows_truth: bool = Field(..., description="Whether the user knows the image's true origin")
    user_label: Literal["ai", "real", "unsure"] = Field(..., description="User's label for the image")
    source_note: Optional[str] = Field(
        None, max_length=500, description="Optional: where the image is from (generator, camera, site)"
    )

    @field_validator("source_note")
    @classmethod
    def _clean_note(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = "".join(ch for ch in v if ch.isprintable() or ch in " \t").strip()
        return cleaned or None


class FeedbackResponse(BaseModel):
    prediction_id: str
    user_knows_truth: bool
    user_label: Literal["ai", "real", "unsure"]
    verdict_correct: Literal["yes", "no", "unsure"] = Field(
        ..., description="Derived server-side by comparing the model verdict with the user's label"
    )


class AdminLoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    password: str = Field(..., min_length=1, max_length=1024)


class AdminLoginResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_at: int = Field(..., description="Unix timestamp (seconds) when the token expires")
