"""
Inference service for running model predictions.
"""

import contextlib
import io
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image

from app.core.config import settings
from app.core.errors import ImageProcessingError, InferenceError, ModelNotFoundError
from app.core.logging import get_logger
from app.services.model_registry import get_model_registry

logger = get_logger(__name__)

# Size of the synthetic image used for the startup warm-up pass. Every
# submodel resizes to its own fixed input size, so this only needs to be a
# valid RGB image.
WARMUP_IMAGE_SIZE = 256


def decode_image(
    image: Optional[Image.Image] = None,
    image_bytes: Optional[bytes] = None,
) -> Optional[Image.Image]:
    """
    Decode the request image once, as an RGB PIL image.

    Matches what each submodel wrapper would otherwise do on its own
    (``Image.open(BytesIO(b)).convert("RGB")``), so outputs are unchanged.
    Returns None when neither argument is given (wrappers then raise their
    usual "No image provided" error).
    """
    if image is not None:
        return image if image.mode == "RGB" else image.convert("RGB")
    if image_bytes is None:
        return None
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ImageProcessingError(
            message="Failed to decode image",
            details={"error": str(e)},
        )


def configure_torch_runtime() -> None:
    """Apply process-wide torch settings that speed up inference on CUDA."""
    if torch.cuda.is_available() and settings.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
        logger.info("phase3 cudnn_benchmark=enabled")


def make_warmup_image(size: int = WARMUP_IMAGE_SIZE) -> Image.Image:
    """Deterministic noise image (avoids degenerate all-constant inputs)."""
    rng = np.random.RandomState(0)
    pixels = rng.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


class InferenceService:
    """
    Service for running inference on individual models.
    """

    def __init__(self):
        self._registry = get_model_registry()

    @staticmethod
    def _inference_context(explain: bool) -> contextlib.AbstractContextManager:
        """
        Grad/precision context for one submodel call.

        The explain path needs autograd (Grad-CAM, attention hooks), so it
        runs exactly as before. The plain path runs under inference_mode and,
        if INFERENCE_FP16 is set and CUDA is available, fp16 autocast.
        """
        if explain:
            return contextlib.nullcontext()
        stack = contextlib.ExitStack()
        stack.enter_context(torch.inference_mode())
        if settings.INFERENCE_FP16 and torch.cuda.is_available():
            stack.enter_context(torch.autocast(device_type="cuda", dtype=torch.float16))
        return stack

    def _run_submodel(
        self,
        submodel,
        image: Optional[Image.Image],
        image_bytes: Optional[bytes],
        explain: bool,
        **kwargs
    ) -> Dict[str, Any]:
        # image_bytes is still passed for wrappers that only accept bytes
        # (dummy models); the real wrappers use the decoded image when given.
        with self._inference_context(explain):
            return submodel.predict(
                image=image, image_bytes=image_bytes, explain=explain, **kwargs
            )

    def predict_single(
        self,
        model_key: str,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        explain: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run prediction on a single submodel.

        Args:
            model_key: Submodel name or repo_id
            image: PIL Image object
            image_bytes: Raw image bytes (alternative to image)
            explain: If True, include explainability heatmap
            **kwargs: Additional arguments for the model

        Returns:
            Standardized prediction dictionary

        Raises:
            ModelNotFoundError: If model not found
            ImageProcessingError: If the image cannot be decoded
            InferenceError: If prediction fails
        """
        try:
            submodel = self._registry.get_submodel(model_key)
            decoded = decode_image(image=image, image_bytes=image_bytes)
            return self._run_submodel(submodel, decoded, image_bytes, explain, **kwargs)
        except (ModelNotFoundError, ImageProcessingError):
            raise
        except Exception as e:
            logger.error(f"Inference failed for {model_key}: {e}")
            raise InferenceError(
                message=f"Inference failed for model {model_key}",
                details={"model": model_key, "error": str(e)}
            )

    def predict_all_submodels(
        self,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        explain: bool = False,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run prediction on all loaded submodels.

        The image is decoded once here and the same RGB PIL image is handed
        to every submodel.

        Args:
            image: PIL Image object
            image_bytes: Raw image bytes (alternative to image)
            explain: If True, include explainability heatmaps
            **kwargs: Additional arguments for the models

        Returns:
            Dictionary mapping submodel name to prediction result

        Raises:
            ImageProcessingError: If the image cannot be decoded
            InferenceError: If any prediction fails
        """
        submodels = self._registry.get_all_submodels()
        decoded = decode_image(image=image, image_bytes=image_bytes)
        results = {}

        for name, submodel in submodels.items():
            try:
                results[name] = self._run_submodel(
                    submodel, decoded, image_bytes, explain, **kwargs
                )
            except Exception as e:
                logger.error(f"Inference failed for submodel {name}: {e}")
                raise InferenceError(
                    message=f"Inference failed for submodel {name}",
                    details={"model": name, "error": str(e)}
                )

        return results

    def warm_up(self) -> Dict[str, float]:
        """
        Run one non-explain forward pass per loaded submodel.

        Moves CUDA context init, lazy kernel loading and cuDNN autotuning off
        the first user request. Failures are logged and skipped: warm-up must
        never prevent the API from serving.

        Returns:
            Mapping of submodel name to warm-up duration in seconds
            (only for submodels that warmed up successfully).
        """
        warmup_t0 = time.perf_counter()
        image = make_warmup_image()
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        durations: Dict[str, float] = {}
        for name, submodel in self._registry.get_all_submodels().items():
            t0 = time.perf_counter()
            try:
                self._run_submodel(submodel, image, image_bytes, explain=False)
            except Exception as e:
                logger.warning(f"Warm-up failed for submodel {name}: {e}")
                continue
            durations[name] = time.perf_counter() - t0
            logger.info(
                "phase3 startup_warmup_model name=%s duration_seconds=%.3f",
                name,
                durations[name],
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        logger.info(
            "phase3 startup_warmup_duration_seconds=%.3f models=%d",
            time.perf_counter() - warmup_t0,
            len(durations),
        )
        return durations


def warm_up_on_startup() -> Optional[Dict[str, float]]:
    """
    Startup hook: configure torch and warm up submodels if enabled.

    Returns per-model warm-up durations, or None when warm-up was skipped
    (WARMUP_ON_STARTUP=false or models not loaded).
    """
    configure_torch_runtime()
    if not settings.WARMUP_ON_STARTUP:
        logger.info("phase3 startup_warmup skipped reason=disabled")
        return None
    if not get_model_registry().is_loaded:
        logger.info("phase3 startup_warmup skipped reason=models_not_loaded")
        return None
    return get_inference_service().warm_up()


# Global singleton instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """
    Get the global inference service instance.

    Returns:
        InferenceService instance
    """
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
