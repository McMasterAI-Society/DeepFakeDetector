"""
Inference service for running model predictions.
"""

from typing import Any, Dict, Optional

from PIL import Image

from app.core.errors import InferenceError, ModelNotFoundError
from app.core.logging import get_logger
from app.services.model_registry import get_model_registry
from app.utils.timing import Timer

logger = get_logger(__name__)


class InferenceService:
    """
    Service for running inference on individual models.
    """
    
    def __init__(self):
        self._registry = get_model_registry()
    
    def predict_single(
        self,
        model_key: str,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run prediction on a single submodel.
        
        Args:
            model_key: Submodel name or repo_id
            image: PIL Image object
            image_bytes: Raw image bytes (alternative to image)
            **kwargs: Additional arguments for the model
            
        Returns:
            Standardized prediction dictionary
            
        Raises:
            ModelNotFoundError: If model not found
            InferenceError: If prediction fails
        """
        try:
            submodel = self._registry.get_submodel(model_key)
            return submodel.predict(image=image, image_bytes=image_bytes, **kwargs)
        except ModelNotFoundError:
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
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run prediction on all loaded submodels.
        
        Args:
            image: PIL Image object
            image_bytes: Raw image bytes (alternative to image)
            **kwargs: Additional arguments for the models
            
        Returns:
            Dictionary mapping submodel name to prediction result
            
        Raises:
            InferenceError: If any prediction fails
        """
        submodels = self._registry.get_all_submodels()
        results = {}
        
        for name, submodel in submodels.items():
            try:
                result = submodel.predict(image=image, image_bytes=image_bytes, **kwargs)
                results[name] = result
            except Exception as e:
                logger.error(f"Inference failed for submodel {name}: {e}")
                raise InferenceError(
                    message=f"Inference failed for submodel {name}",
                    details={"model": name, "error": str(e)}
                )
        
        return results


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
