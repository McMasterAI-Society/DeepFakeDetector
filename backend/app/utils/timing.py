"""
Timing utilities for performance measurement.
"""

import time
from contextlib import contextmanager
from typing import Dict, Generator, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Timer class for measuring execution time of code blocks.
    
    Usage:
        timer = Timer()
        with timer.measure("inference"):
            # do inference
        with timer.measure("fusion"):
            # do fusion
        timings = timer.get_timings()
    """
    
    def __init__(self):
        self._start_time: Optional[float] = None
        self._timings: Dict[str, int] = {}
        self._total_start: Optional[float] = None
    
    def start_total(self) -> None:
        """Start the total timer."""
        self._total_start = time.perf_counter()
    
    def stop_total(self) -> None:
        """Stop the total timer and record the duration."""
        if self._total_start is not None:
            elapsed_ms = int((time.perf_counter() - self._total_start) * 1000)
            self._timings["total"] = elapsed_ms
    
    @contextmanager
    def measure(self, name: str) -> Generator[None, None, None]:
        """
        Context manager to measure execution time of a block.
        
        Args:
            name: Name for this timing measurement
            
        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            self._timings[name] = elapsed_ms
            logger.debug(f"Timer [{name}]: {elapsed_ms}ms")
    
    def record(self, name: str, duration_ms: int) -> None:
        """
        Manually record a timing.
        
        Args:
            name: Name for this timing
            duration_ms: Duration in milliseconds
        """
        self._timings[name] = duration_ms
    
    def get_timings(self) -> Dict[str, int]:
        """
        Get all recorded timings.
        
        Returns:
            Dictionary of timing name -> milliseconds
        """
        return self._timings.copy()
    
    def get(self, name: str) -> Optional[int]:
        """
        Get a specific timing.
        
        Args:
            name: Timing name
            
        Returns:
            Duration in milliseconds, or None if not recorded
        """
        return self._timings.get(name)
    
    def reset(self) -> None:
        """Reset all timings."""
        self._timings.clear()
        self._total_start = None


def measure_time(func):
    """
    Decorator to measure function execution time.
    
    Logs the execution time at DEBUG level.
    """
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            logger.debug(f"Function [{func.__name__}]: {elapsed_ms}ms")
    return wrapper
