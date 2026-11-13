"""
User feedback route: "Do you know whether this image is AI-generated?".
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException

from app.core.logging import get_logger
from app.schemas.feedback import FeedbackRequest, FeedbackResponse
from app.services.rate_limit import rate_limit
from app.services.store import get_store

logger = get_logger(__name__)
router = APIRouter(tags=["feedback"])


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit feedback on a prediction",
    description="One feedback entry per prediction; submitting again replaces it. "
                "verdict_correct is derived server-side from the stored verdict.",
    responses={404: {"description": "Unknown prediction_id"}, 429: {"description": "Rate limit exceeded"},
               503: {"description": "Prediction store unavailable"}},
    dependencies=[Depends(rate_limit("feedback"))],
)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    store = await asyncio.to_thread(get_store)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "StoreUnavailable", "message": "Feedback is temporarily unavailable"},
        )
    try:
        result = await asyncio.to_thread(
            store.upsert_feedback,
            prediction_id=body.prediction_id,
            user_knows_truth=body.user_knows_truth,
            user_label=body.user_label,
            source_note=body.source_note,
        )
    except Exception as e:
        logger.error(f"Failed to save feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "FeedbackError", "message": "Could not save feedback"},
        )
    if result is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "NotFound", "message": "Unknown prediction_id"},
        )
    return FeedbackResponse(**result)
