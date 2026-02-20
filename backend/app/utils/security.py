"""
Security utilities for the application.
"""

import hashlib
import secrets
from typing import List, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Maximum allowed file size (10 MB)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

# Allowed image MIME types
ALLOWED_MIME_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/bmp"
]

# Image magic bytes for validation
IMAGE_SIGNATURES = {
    b'\xff\xd8\xff': 'image/jpeg',
    b'\x89PNG\r\n\x1a\n': 'image/png',
    b'GIF87a': 'image/gif',
    b'GIF89a': 'image/gif',
    b'RIFF': 'image/webp',  # WebP (partial)
    b'BM': 'image/bmp'
}


def validate_file_size(content: bytes, max_size: int = MAX_FILE_SIZE_BYTES) -> bool:
    """
    Validate that file size is within allowed limits.
    
    Args:
        content: File content as bytes
        max_size: Maximum allowed size in bytes
        
    Returns:
        True if valid, False otherwise
    """
    return len(content) <= max_size


def detect_mime_type(content: bytes) -> Optional[str]:
    """
    Detect MIME type from file content using magic bytes.
    
    Args:
        content: File content as bytes
        
    Returns:
        Detected MIME type or None
    """
    for signature, mime_type in IMAGE_SIGNATURES.items():
        if content.startswith(signature):
            return mime_type
    return None


def validate_image_content(
    content: bytes,
    allowed_types: List[str] = ALLOWED_MIME_TYPES
) -> bool:
    """
    Validate image content by checking magic bytes.
    
    Args:
        content: File content as bytes
        allowed_types: List of allowed MIME types
        
    Returns:
        True if valid image type, False otherwise
    """
    detected_type = detect_mime_type(content)
    if detected_type is None:
        return False
    return detected_type in allowed_types


def compute_file_hash(content: bytes, algorithm: str = "sha256") -> str:
    """
    Compute hash of file content.
    
    Args:
        content: File content as bytes
        algorithm: Hash algorithm (sha256, md5, etc.)
        
    Returns:
        Hex-encoded hash string
    """
    hasher = hashlib.new(algorithm)
    hasher.update(content)
    return hasher.hexdigest()


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    
    Returns:
        Random hex string
    """
    return secrets.token_hex(8)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Remove path separators and null bytes
    sanitized = filename.replace("/", "_").replace("\\", "_").replace("\x00", "")
    # Remove leading dots to prevent hidden files
    sanitized = sanitized.lstrip(".")
    return sanitized[:255] if sanitized else "unnamed"
