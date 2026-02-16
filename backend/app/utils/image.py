"""
Image processing utilities.
"""

from io import BytesIO
from typing import Optional, Tuple

from PIL import Image

from app.core.errors import ImageProcessingError
from app.core.logging import get_logger

logger = get_logger(__name__)


def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """
    Load a PIL Image from raw bytes.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        PIL Image object
        
    Raises:
        ImageProcessingError: If image cannot be decoded
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise ImageProcessingError(
            message="Failed to decode image",
            details={"error": str(e)}
        )


def validate_image_bytes(image_bytes: bytes) -> bool:
    """
    Validate that bytes represent a valid image.
    
    Args:
        image_bytes: Raw image bytes
        
    Returns:
        True if valid image, False otherwise
    """
    try:
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        return True
    except Exception:
        return False


def get_image_info(image: Image.Image) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image: PIL Image object
        
    Returns:
        Dictionary with image info
    """
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format
    }


def resize_image(
    image: Image.Image,
    size: Tuple[int, int],
    resample: int = Image.Resampling.LANCZOS
) -> Image.Image:
    """
    Resize image to specified size.
    
    Args:
        image: PIL Image object
        size: Target (width, height)
        resample: Resampling filter
        
    Returns:
        Resized PIL Image
    """
    return image.resize(size, resample=resample)


def image_to_bytes(
    image: Image.Image,
    format: str = "PNG",
    quality: int = 95
) -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image object
        format: Output format (PNG, JPEG, etc.)
        quality: JPEG quality (1-95)
        
    Returns:
        Image as bytes
    """
    buffer = BytesIO()
    if format.upper() == "JPEG":
        image.save(buffer, format=format, quality=quality)
    else:
        image.save(buffer, format=format)
    return buffer.getvalue()
