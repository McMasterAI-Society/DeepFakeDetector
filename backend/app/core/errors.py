"""
Custom exceptions and error handling for the application.
"""

from typing import Any, Dict, Optional


class DeepFakeDetectorError(Exception):
    """Base exception for DeepFake Detector application."""
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ModelNotLoadedError(DeepFakeDetectorError):
    """Raised when attempting to use a model that hasn't been loaded."""
    pass


class ModelNotFoundError(DeepFakeDetectorError):
    """Raised when a requested model is not found in the registry."""
    pass


class HuggingFaceDownloadError(DeepFakeDetectorError):
    """Raised when downloading from Hugging Face fails."""
    pass


class ImageProcessingError(DeepFakeDetectorError):
    """Raised when image processing/decoding fails."""
    pass


class InferenceError(DeepFakeDetectorError):
    """Raised when model inference fails."""
    pass


class FusionError(DeepFakeDetectorError):
    """Raised when fusion prediction fails."""
    pass


class ConfigurationError(DeepFakeDetectorError):
    """Raised when configuration is invalid or missing."""
    pass
