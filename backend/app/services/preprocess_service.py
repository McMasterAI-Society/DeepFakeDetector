"""
Image preprocessing service.
"""

from typing import Optional, Dict, Any

from PIL import Image

from app.core.errors import ImageProcessingError
from app.core.logging import get_logger
from app.utils.image import load_image_from_bytes, validate_image_bytes
from app.utils.security import validate_file_size, validate_image_content, MAX_FILE_SIZE_BYTES

logger = get_logger(__name__)


class PreprocessService:
    """
    Service for preprocessing images before model inference.
    
    For Milestone 1, this is minimal - just validates and optionally
    decodes images. Future milestones will add more preprocessing.
    """
    
    def __init__(self, max_file_size: int = MAX_FILE_SIZE_BYTES):
        """
        Initialize the preprocess service.
        
        Args:
            max_file_size: Maximum allowed file size in bytes
        """
        self.max_file_size = max_file_size
    
    def validate_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Validate uploaded image bytes.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Dictionary with validation results
            
        Raises:
            ImageProcessingError: If validation fails
        """
        # Check file size
        if not validate_file_size(image_bytes, self.max_file_size):
            raise ImageProcessingError(
                message=f"File too large. Maximum size is {self.max_file_size // (1024*1024)}MB",
                details={"size": len(image_bytes), "max_size": self.max_file_size}
            )
        
        # Check content type via magic bytes
        if not validate_image_content(image_bytes):
            raise ImageProcessingError(
                message="Invalid image format. Supported formats: JPEG, PNG, GIF, WebP, BMP",
                details={"size": len(image_bytes)}
            )
        
        return {
            "valid": True,
            "size_bytes": len(image_bytes)
        }
    
    def decode_image(self, image_bytes: bytes) -> Image.Image:
        """
        Decode image bytes to PIL Image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            PIL Image object
            
        Raises:
            ImageProcessingError: If decoding fails
        """
        return load_image_from_bytes(image_bytes)
    
    def preprocess(
        self,
        image_bytes: bytes,
        decode: bool = False
    ) -> Dict[str, Any]:
        """
        Full preprocessing pipeline.
        
        Args:
            image_bytes: Raw image bytes
            decode: Whether to decode to PIL Image
            
        Returns:
            Dictionary with:
            - image_bytes: Original or processed bytes
            - image: PIL Image if decode=True
            - validation: Validation results
        """
        # Validate
        validation = self.validate_image(image_bytes)
        
        result = {
            "image_bytes": image_bytes,
            "validation": validation
        }
        
        # Optionally decode
        if decode:
            result["image"] = self.decode_image(image_bytes)
        
        return result


# Global singleton instance
_preprocess_service: Optional[PreprocessService] = None


def get_preprocess_service() -> PreprocessService:
    """
    Get the global preprocess service instance.
    
    Returns:
        PreprocessService instance
    """
    global _preprocess_service
    if _preprocess_service is None:
        _preprocess_service = PreprocessService()
    return _preprocess_service
