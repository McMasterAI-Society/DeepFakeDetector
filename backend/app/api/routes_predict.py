"""
Prediction routes.
"""

from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

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
    ErrorResponse
)
from app.services.inference_service import get_inference_service
from app.services.fusion_service import get_fusion_service
from app.services.preprocess_service import get_preprocess_service
from app.services.model_registry import get_model_registry
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
                    image_bytes=image_bytes
                )
            
            # Run fusion
            with timer.measure("fusion"):
                final_result = fusion_service.fuse(submodel_outputs=submodel_outputs)
            
            timer.stop_total()
            
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
                        prob_fake=output["prob_fake"]
                    )
                    for name, output in submodel_outputs.items()
                } if should_return_submodels else None,
                timing_ms=TimingInfo(**timer.get_timings())
            )
        
        else:
            # Single model prediction
            model_key = model or registry.get_submodel_names()[0]
            
            with timer.measure("inference"):
                result = inference_service.predict_single(
                    model_key=model_key,
                    image_bytes=image_bytes
                )
            
            timer.stop_total()
            
            return PredictResponse(
                final=PredictionResult(
                    pred=result["pred"],
                    pred_int=result["pred_int"],
                    prob_fake=result["prob_fake"]
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
