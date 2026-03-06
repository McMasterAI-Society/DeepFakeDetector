"""
Prediction routes.
"""

import base64
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile

from app.core.errors import (
    DeepFakeDetectorError,
    ImageProcessingError,
    InferenceError,
    FusionError,
    ModelNotFoundError,
    ModelNotLoadedError
)
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
from app.utils.timing import Timer

logger = get_logger(__name__)
router = APIRouter(tags=["predict"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict if image is real or fake",
    description="Upload an image to get a deepfake detection prediction",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or request"},
        404: {"model": ErrorResponse, "description": "Model not found"},
        500: {"model": ErrorResponse, "description": "Inference error"}
    }
)
async def predict(
    image: UploadFile = File(..., description="Image file to analyze"),
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
            return PredictResponse(
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
            
            return PredictResponse(
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


@router.post("/explain-model", response_model=ExplainModelResponse)
async def explain_model(
    image: UploadFile = File(...),
    model_name: str = Form(...),
    prob_fake: float = Form(...),
    contribution_percentage: float = Form(None),
    heatmap_base64: str = Form(None),
    focus_summary: str = Form(None)
):
    """
    Generate an on-demand LLM explanation for a single model's prediction.
    This endpoint is token-efficient - only called when user requests insights.
    """
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
        
        return ExplainModelResponse(
            model_name=model_name,
            insight=SingleModelInsight(
                key_finding=result["key_finding"],
                what_model_saw=result["what_model_saw"],
                important_regions=result["important_regions"],
                confidence_qualifier=result["confidence_qualifier"]
            )
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating model explanation: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "ExplanationError", "message": str(e)}
        )
