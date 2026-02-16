"""
Cache service for managing cached data.

Note: For Milestone 1, this is a simple placeholder.
Future milestones may add more sophisticated caching.
"""

from typing import Any, Dict, Optional
import hashlib
import time

from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheService:
    """
    Simple in-memory cache service.
    
    This is a basic implementation for Milestone 1.
    Future versions may use Redis or other backends.
    """
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """
        Initialize the cache service.
        
        Args:
            max_size: Maximum number of items to cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def _generate_key(self, data: bytes) -> str:
        """Generate a cache key from data (SHA256 hash)."""
        return hashlib.sha256(data).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        entry = self._cache.get(key)
        if entry is None:
            return None
        
        # Check TTL
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self._cache[key]
            return None
        
        return entry["value"]
    
    def set(self, key: str, value: Any) -> None:
        """
        Set a value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        # Simple eviction: remove oldest if at max size
        if len(self._cache) >= self.max_size:
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
        
        self._cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)


# Global singleton instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """
    Get the global cache service instance.
    
    Returns:
        CacheService instance
    """
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
