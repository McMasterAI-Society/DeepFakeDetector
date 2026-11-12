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
from app.models.wrappers.foundation_probe_wrapper import BACKBONE_DISPLAY_KEYS
from app.utils.timing import Timer

logger = get_logger(__name__)
router = APIRouter(tags=["predict"])


def _legacy_model_version(registry) -> str:
    manifest_hash = registry.get_manifest_hash()
    return f"legacy-fusion:{manifest_hash[:12]}" if manifest_hash else "legacy-fusion"


def _legacy_verdict(pred: str) -> str:
    return "ai" if pred == "fake" else "real"


def _run_probe(image_bytes: bytes, explain: bool, return_submodels: bool, timer: Timer) -> PredictResponse:
    """Run the foundation probe and build the response (timer already started)."""
    probe = get_model_registry().get_probe()
    with timer.measure("inference"):
        result = get_inference_service().predict_single(
            model_key=probe.name, image_bytes=image_bytes, explain=explain
        )
    timer.stop_total()
    return _probe_response(result, timer, return_submodels)


def _probe_response(
    result: dict,
    timer: Timer,
    return_submodels: bool,
) -> PredictResponse:
    """
    Build the /predict response for the foundation probe.

    The two backbones are reported as "submodels" so the existing frontend tabs
    work: prob_fake is the calibrated probability with only that backbone's
    evidence (the other at its training mean), contribution_percentage is the
    backbone's share of |logit| and fusion_meta.weighted_contributions holds the
    signed logit contributions (they sum, with logit_intercept, to the raw logit).
    """
    backbones = result.get("backbones", {})
    names = {key: BACKBONE_DISPLAY_KEYS.get(key, key) for key in backbones}
    fusion_meta = FusionMeta(
        submodel_weights={names[k]: v["coef_norm"] for k, v in backbones.items()},
        weighted_contributions={names[k]: v["logit_contribution"] for k, v in backbones.items()},
        contribution_percentages={names[k]: v["contribution_percentage"] for k, v in backbones.items()},
        logit_intercept=result.get("meta", {}).get("intercept"),
    )
    submodels = None
    display_info = None
    if return_submodels:
        submodels = {
            names[k]: PredictionResult(
                pred=v["pred"],
                pred_int=v["pred_int"],
                prob_fake=v["prob_fake"],
                heatmap_base64=v.get("heatmap_base64"),
                explainability_type=v.get("explainability_type"),
                focus_summary=v.get("focus_summary"),
                contribution_percentage=v["contribution_percentage"],
            )
            for k, v in backbones.items()
        }
        display_info = {n: ModelDisplayInfo(**get_model_display_info(n)) for n in submodels}
    return PredictResponse(
        final=PredictionResult(
            pred=result["pred"],
            pred_int=result["pred_int"],
            prob_fake=result["prob_fake"],
            heatmap_base64=result.get("heatmap_base64"),
            explainability_type=result.get("explainability_type"),
            focus_summary=result.get("focus_summary"),
        ),
        fusion_used=False,
        submodels=submodels,
        fusion_meta=fusion_meta,
        model_display_info=display_info,
        timing_ms=TimingInfo(**timer.get_timings()),
        verdict=result["verdict"],
        verdict_bands=result.get("meta", {}).get("verdict_bands"),
        model_version=result.get("model_version"),
    )


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

    In foundation-probe mode (MODEL_MODE, the default) the probe is the only
    model: use_fusion is ignored, the two backbones are reported as submodels
    with their logit contributions, and ``verdict`` can be "inconclusive".
    
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

        if registry.mode == "foundation-probe":
            if model and registry.get_submodel(model) is not registry.get_probe():
                raise ModelNotFoundError(message=f"Submodel not found: {model}")
            return _run_probe(
                image_bytes,
                explain=explain and settings.PROBE_EXPLAIN_ON_PREDICT,
                return_submodels=return_submodels if return_submodels is not None else True,
                timer=timer,
            )
        
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
                timing_ms=TimingInfo(**timer.get_timings()),
                verdict=_legacy_verdict(final_result["pred"]),
                model_version=_legacy_model_version(registry),
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
                timing_ms=TimingInfo(**timer.get_timings()),
                verdict=_legacy_verdict(result["pred"]),
                model_version=_legacy_model_version(registry),
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


@router.post(
    "/explain",
    response_model=PredictResponse,
    summary="Prediction with heatmaps (foundation probe)",
    description="Same as /predict but always computes the probe's input-gradient heatmaps. "
                "Slower (~2.8x), so the frontend calls it on demand.",
    responses={409: {"model": ErrorResponse, "description": "Not in foundation-probe mode"}},
)
async def explain_probe(
    image: UploadFile = File(..., description="Image file to analyze"),
    return_submodels: bool = Query(True, description="Include per-backbone results"),
) -> PredictResponse:
    timer = Timer()
    timer.start_total()
    try:
        registry = get_model_registry()
        if registry.mode != "foundation-probe":
            raise HTTPException(
                status_code=409,
                detail={"error": "NotSupported",
                        "message": "/explain needs MODEL_MODE=foundation-probe; use /predict?explain=true"},
            )
        with timer.measure("download"):
            image_bytes = await image.read()
        with timer.measure("preprocess"):
            get_preprocess_service().validate_image(image_bytes)
        return _run_probe(image_bytes, explain=True, return_submodels=return_submodels, timer=timer)
    except HTTPException:
        raise
    except ImageProcessingError as e:
        raise HTTPException(status_code=400, detail={"error": "ImageProcessingError", "message": e.message})
    except ModelNotLoadedError as e:
        raise HTTPException(status_code=503, detail={"error": "ModelNotLoadedError", "message": e.message})
    except InferenceError as e:
        raise HTTPException(status_code=500, detail={"error": "InferenceError", "message": e.message})
    except Exception as e:
        logger.exception(f"Unexpected error in explain endpoint: {e}")
        raise HTTPException(status_code=500, detail={"error": "InternalError", "message": str(e)})


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
