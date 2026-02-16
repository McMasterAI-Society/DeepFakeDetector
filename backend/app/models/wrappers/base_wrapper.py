"""
Base wrapper class for model wrappers.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from PIL import Image


class BaseModelWrapper(ABC):
    """
    Abstract base class for model wrappers.
    
    All model wrappers should inherit from this class and implement
    the abstract methods.
    """
    
    def __init__(
        self,
        repo_id: str,
        config: Dict[str, Any],
        local_path: str
    ):
        """
        Initialize the wrapper.
        
        Args:
            repo_id: Hugging Face repository ID
            config: Configuration from config.json
            local_path: Local path where the model files are stored
        """
        self.repo_id = repo_id
        self.config = config
        self.local_path = local_path
        self._predict_fn: Optional[Callable] = None
    
    @property
    def name(self) -> str:
        """Get the short name of the model (last part of repo_id)."""
        return self.repo_id.split("/")[-1]
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model and prepare for inference.
        
        This method should import the predict function from the downloaded
        repository and store it for later use.
        """
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Run prediction.
        
        Returns:
            Dictionary with standardized prediction fields:
            - pred_int: 0 (real) or 1 (fake)
            - pred: "real" or "fake"
            - prob_fake: float probability
            - meta: dict with any additional metadata
        """
        pass
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        return self._predict_fn is not None
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model info
        """
        return {
            "repo_id": self.repo_id,
            "name": self.name,
            "config": self.config,
            "local_path": self.local_path,
            "is_loaded": self.is_loaded()
        }


class BaseSubmodelWrapper(BaseModelWrapper):
    """Base wrapper for submodels that process images."""
    
    @abstractmethod
    def predict(
        self,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run prediction on an image.
        
        Args:
            image: PIL Image object
            image_bytes: Raw image bytes (alternative to image)
            **kwargs: Additional arguments
            
        Returns:
            Standardized prediction dictionary
        """
        pass


class BaseFusionWrapper(BaseModelWrapper):
    """Base wrapper for fusion models that combine submodel outputs."""
    
    @abstractmethod
    def predict(
        self,
        submodel_outputs: Dict[str, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run fusion prediction on submodel outputs.
        
        Args:
            submodel_outputs: Dictionary mapping submodel name to its output
            **kwargs: Additional arguments
            
        Returns:
            Standardized prediction dictionary
        """
        pass
