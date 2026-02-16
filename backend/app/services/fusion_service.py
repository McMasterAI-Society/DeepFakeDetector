"""
Fusion service for combining submodel predictions.
"""

from typing import Any, Dict

from app.core.errors import FusionError
from app.core.logging import get_logger
from app.services.model_registry import get_model_registry

logger = get_logger(__name__)


class FusionService:
    """
    Service for running fusion predictions.
    """
    
    def __init__(self):
        self._registry = get_model_registry()
    
    def fuse(
        self,
        submodel_outputs: Dict[str, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run fusion on submodel outputs.
        
        Args:
            submodel_outputs: Dictionary mapping submodel name to its prediction output
            **kwargs: Additional arguments for the fusion model
            
        Returns:
            Standardized prediction dictionary
            
        Raises:
            FusionError: If fusion fails
        """
        try:
            fusion = self._registry.get_fusion()
            return fusion.predict(submodel_outputs=submodel_outputs, **kwargs)
        except FusionError:
            raise
        except Exception as e:
            logger.error(f"Fusion failed: {e}")
            raise FusionError(
                message="Fusion prediction failed",
                details={"error": str(e)}
            )


# Global singleton instance
_fusion_service = None


def get_fusion_service() -> FusionService:
    """
    Get the global fusion service instance.
    
    Returns:
        FusionService instance
    """
    global _fusion_service
    if _fusion_service is None:
        _fusion_service = FusionService()
    return _fusion_service
