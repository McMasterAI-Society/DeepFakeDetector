"""
Prediction routes.
"""

import asyncio
import base64
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, Request, UploadFile
)

from app.core.errors import (
    DeepFakeDetectorError,
    ImageProcessingError,
    InferenceError,
    FusionError,
    ModelNotFoundError,
    ModelNotLoadedError
)
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.predict import (
    PredictResponse,
    PredictionResult,
    TimingInfo,
    ErrorResponse,
    FusionMeta,
    ModelDisplayInfo,
    ExplainModelResponse,
    SingleModelInsight
)
from app.services.inference_service import get_inference_service
from app.services.fusion_service import get_fusion_service
from app.services.preprocess_service import get_preprocess_service
from app.services.model_registry import get_model_registry
from app.services.llm_service import get_llm_service, get_model_display_info, MODEL_DISPLAY_INFO
from app.services.rate_limit import get_client_ip, rate_limit
from app.services.store import get_store
from app.utils.timing import Timer

logger = get_logger(__name__)
router = APIRouter(tags=["predict"])

_EXPLAIN_KEYS = ("heatmap_base64", "explainability_type", "focus_summary")


def _manifest_id() -> Optional[str]:
    """Identifier of the serving model set, recorded with each prediction."""
    try:
        registry = get_model_registry()
        manifest_hash = getattr(registry, "manifest_hash", None)
        if isinstance(manifest_hash, str) and manifest_hash:
            return manifest_hash
        return registry.get_fusion_repo_id()
    except Exception:
        return None


async def _persist_prediction(
    *,
    request: Request,
    background_tasks: BackgroundTasks,
    image_bytes: bytes,
    final: Dict[str, Any],
    outputs: Dict[str, Dict[str, Any]],
    contributions: Dict[str, float],
    fusion_used: bool,
    consent_store_image: bool,
    latency_ms: Optional[int],
) -> Optional[str]:
    """
    Record prediction metadata (always) and queue the consented image write.
    Never raises: a store failure must not break /predict.
    """
    try:
        store = await asyncio.to_thread(get_store)
        if store is None:
            return None
        submodels = {
            name: {
                "pred": out.get("pred"),
                "prob_fake": out.get("prob_fake"),
                "contribution_percentage": contributions.get(name),
            }
            for name, out in outputs.items()
        }
        # Heatmaps can reveal the outline of the subject, so they are stored
        # only with the same consent as the image itself.
        explain_keys = (
            _EXPLAIN_KEYS
            if consent_store_image
            else tuple(k for k in _EXPLAIN_KEYS if k != "heatmap_base64")
        )
        explain = {
            name: {k: out[k] for k in explain_keys if out.get(k) is not None}
            for name, out in outputs.items()
        }
        prediction_id = await asyncio.to_thread(
            store.record_prediction,
            image_bytes=image_bytes,
            final_pred=final["pred"],
            final_prob=final["prob_fake"],
            fusion_used=fusion_used,
            submodels=submodels,
            explain=explain,
            variant=settings.PREDICTION_VARIANT,
            latency_ms=latency_ms,
            client_ip=get_client_ip(request),
            manifest_hash=_manifest_id(),
            consent_store_image=consent_store_image,
        )
        if consent_store_image:
            # Runs after the response is sent (in the threadpool).
            background_tasks.add_task(store.save_consented_image, prediction_id, image_bytes)
        return prediction_id
    except Exception as e:
        logger.warning(f"Failed to record prediction: {e}")
        return None


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict if image is real or fake",
    description="Upload an image to get a deepfake detection prediction",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Inference error"}
    },
    dependencies=[Depends(rate_limit("predict"))],
)
async def predict(
    request: Request,
    background_tasks: BackgroundTasks,
    image: UploadFile = File(..., description="Image file to analyze"),
    consent_store_image: bool = Form(
        False,
        description="User explicitly consents to storing this image for research. "
                    "When false (default) no image bytes are written to disk."
    ),
    use_fusion: bool = Query(
        True,
        description="Use fusion model (majority vote) across all submodels"
    ),
    model: Optional[str] = Query(
        None,
        description="Specific submodel to use (name or repo_id). Only used when use_fusion=false"
    ),
    return_submodels: Optional[bool] = Query(
        None,
        description="Include individual submodel predictions in response. Defaults to true when use_fusion=true"
    ),
    explain: bool = Query(
        True,
        description="Generate explainability heatmaps (Grad-CAM for CNNs, attention rollout for transformers)"
    )
) -> PredictResponse:
    """
    Predict if an uploaded image is real or fake.
    
    When use_fusion=true (default):
    - Runs all submodels on the image
    - Combines predictions using majority vote fusion
    - Returns the fused result plus optionally individual submodel results
    
    When use_fusion=false:
    - Runs only the specified submodel (or the first available if not specified)
    - Returns just that model's prediction
    
    Response includes timing information for each step.
    """
    timer = Timer()
    timer.start_total()
    
    # Determine if we should return submodel results
    should_return_submodels = return_submodels if return_submodels is not None else use_fusion
    
    try:
        # Read image bytes
        with timer.measure("download"):
            image_bytes = await image.read()
        
        # Validate and preprocess
        with timer.measure("preprocess"):
            preprocess_service = get_preprocess_service()
            preprocess_service.validate_image(image_bytes)
        
        inference_service = get_inference_service()
        fusion_service = get_fusion_service()
        registry = get_model_registry()
        
        if use_fusion:
            # Run all submodels
            with timer.measure("inference"):
                submodel_outputs = inference_service.predict_all_submodels(
                    image_bytes=image_bytes,
                    explain=explain
                )
            
            # Run fusion
            with timer.measure("fusion"):
                final_result = fusion_service.fuse(submodel_outputs=submodel_outputs)
            
            timer.stop_total()
            
            # Extract fusion meta (contribution percentages)
            fusion_meta_dict = final_result.get("meta", {})
            contribution_percentages = fusion_meta_dict.get("contribution_percentages", {})
            
            # Build fusion meta object
            fusion_meta = FusionMeta(
                submodel_weights=fusion_meta_dict.get("submodel_weights", {}),
                weighted_contributions=fusion_meta_dict.get("weighted_contributions", {}),
                contribution_percentages=contribution_percentages
            ) if fusion_meta_dict else None
            
            # Build model display info for frontend
            model_display_info = {
                name: ModelDisplayInfo(**get_model_display_info(name))
                for name in submodel_outputs.keys()
            }
            
            # Build response
            response = PredictResponse(
                final=PredictionResult(
                    pred=final_result["pred"],
                    pred_int=final_result["pred_int"],
                    prob_fake=final_result["prob_fake"]
                ),
                fusion_used=True,
                submodels={
                    name: PredictionResult(
                        pred=output["pred"],
                        pred_int=output["pred_int"],
                        prob_fake=output["prob_fake"],
                        heatmap_base64=output.get("heatmap_base64"),
                        explainability_type=output.get("explainability_type"),
                        focus_summary=output.get("focus_summary"),
                        contribution_percentage=contribution_percentages.get(name)
                    )
                    for name, output in submodel_outputs.items()
                } if should_return_submodels else None,
                fusion_meta=fusion_meta,
                model_display_info=model_display_info if should_return_submodels else None,
                timing_ms=TimingInfo(**timer.get_timings())
            )
            response.prediction_id = await _persist_prediction(
                request=request,
                background_tasks=background_tasks,
                image_bytes=image_bytes,
                final=final_result,
                outputs=submodel_outputs,
                contributions=contribution_percentages,
                fusion_used=True,
                consent_store_image=consent_store_image,
                latency_ms=timer.get_timings().get("total"),
            )
            return response
        
        else:
            # Single model prediction
            model_key = model or registry.get_submodel_names()[0]
            
            with timer.measure("inference"):
                result = inference_service.predict_single(
                    model_key=model_key,
                    image_bytes=image_bytes,
                    explain=explain
                )
            
            timer.stop_total()
            
            response = PredictResponse(
                final=PredictionResult(
                    pred=result["pred"],
                    pred_int=result["pred_int"],
                    prob_fake=result["prob_fake"],
                    heatmap_base64=result.get("heatmap_base64"),
                    explainability_type=result.get("explainability_type"),
                    focus_summary=result.get("focus_summary")
                ),
                fusion_used=False,
                submodels=None,
                timing_ms=TimingInfo(**timer.get_timings())
            )
            response.prediction_id = await _persist_prediction(
                request=request,
                background_tasks=background_tasks,
                image_bytes=image_bytes,
                final=result,
                outputs={model_key: result},
                contributions={},
                fusion_used=False,
                consent_store_image=consent_store_image,
                latency_ms=timer.get_timings().get("total"),
            )
            return response
    
    except ImageProcessingError as e:
        logger.warning(f"Image processing error: {e.message}")
        raise HTTPException(
            status_code=400,
            detail={"error": "ImageProcessingError", "message": e.message, "details": e.details}
        )
    
    except ModelNotFoundError as e:
        logger.warning(f"Model not found: {e.message}")
        raise HTTPException(
            status_code=404,
            detail={"error": "ModelNotFoundError", "message": e.message, "details": e.details}
        )
    
    except ModelNotLoadedError as e:
        logger.error(f"Models not loaded: {e.message}")
        raise HTTPException(
            status_code=503,
            detail={"error": "ModelNotLoadedError", "message": e.message, "details": e.details}
        )
    
    except (InferenceError, FusionError) as e:
        logger.error(f"Inference/Fusion error: {e.message}")
        raise HTTPException(
            status_code=500,
            detail={"error": type(e).__name__, "message": e.message, "details": e.details}
        )
    
    except Exception as e:
        logger.exception(f"Unexpected error in predict endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "InternalError", "message": str(e)}
        )


@router.post(
    "/explain-model",
    response_model=ExplainModelResponse,
    dependencies=[Depends(rate_limit("explain"))],
)
async def explain_model(
    image: UploadFile = File(...),
    model_name: str = Form(...),
    prob_fake: float = Form(...),
    contribution_percentage: float = Form(None),
    heatmap_base64: str = Form(None),
    focus_summary: str = Form(None),
    prediction_id: Optional[str] = Form(None, max_length=64),
):
    """
    Generate an on-demand LLM explanation for a single model's prediction.
    This endpoint is token-efficient - only called when user requests insights.

    When prediction_id is given and known, the stored server-side values for
    the model (probability, heatmap, focus summary, contribution) replace the
    client-supplied ones, and the generated insight is saved with the record.
    """
    store = None
    if prediction_id:
        record = None
        try:
            store = await asyncio.to_thread(get_store)
            if store is not None:
                record = await asyncio.to_thread(store.get_prediction, prediction_id)
        except Exception as e:
            logger.warning(f"Prediction lookup failed for explain-model: {e}")
        stored_sub = (record or {}).get("submodels", {}).get(model_name)
        if stored_sub is not None:
            stored_explain = record.get("explain", {}).get(model_name, {})
            if stored_sub.get("prob_fake") is not None:
                prob_fake = stored_sub["prob_fake"]
            if stored_sub.get("contribution_percentage") is not None:
                contribution_percentage = stored_sub["contribution_percentage"]
            heatmap_base64 = stored_explain.get("heatmap_base64", heatmap_base64)
            focus_summary = stored_explain.get("focus_summary", focus_summary)
        else:
            store = None

    try:
        # Read and validate image
        image_bytes = await image.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Encode image to base64 for LLM
        original_b64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Get LLM service
        llm_service = get_llm_service()
        if not llm_service.enabled:
            raise HTTPException(
                status_code=503, 
                detail="LLM service is not enabled. Set GEMINI_API_KEY environment variable."
            )
        
        # Generate explanation
        result = llm_service.generate_single_model_explanation(
            model_name=model_name,
            original_image_b64=original_b64,
            prob_fake=prob_fake,
            heatmap_b64=heatmap_base64,
            contribution_percentage=contribution_percentage,
            focus_summary=focus_summary
        )
        
        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate explanation from LLM"
            )
        
        insight = SingleModelInsight(
            key_finding=result["key_finding"],
            what_model_saw=result["what_model_saw"],
            important_regions=result["important_regions"],
            confidence_qualifier=result["confidence_qualifier"]
        )
        if store is not None and prediction_id:
            try:
                await asyncio.to_thread(
                    store.add_llm_insight, prediction_id, model_name, insight.model_dump()
                )
            except Exception as e:
                logger.warning(f"Failed to save LLM insight: {e}")

        return ExplainModelResponse(model_name=model_name, insight=insight)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating model explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "ExplanationError", "message": str(e)}
        )
