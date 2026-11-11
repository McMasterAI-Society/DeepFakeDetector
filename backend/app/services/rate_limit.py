"""
In-process token-bucket rate limiting keyed by client IP.

Limits are per process: with several workers the effective limit is
multiplied by the worker count. Configure via RATE_LIMIT_* settings.
"""

from __future__ import annotations

import ipaddress
import math
import threading
import time
from collections import OrderedDict
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

from fastapi import HTTPException, Request

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_PERIODS = {"second": 1, "sec": 1, "s": 1, "minute": 60, "min": 60, "m": 60,
            "hour": 3600, "h": 3600, "day": 86400, "d": 86400}


@lru_cache(maxsize=64)
def parse_rate(spec: Optional[str]) -> Optional[Tuple[float, float]]:
    """Parse "<count>/<period>" into (capacity, period_seconds). None means unlimited."""
    if not spec or spec.strip().lower() in ("0", "off", "none", "unlimited"):
        return None
    count_s, _, period_s = spec.strip().partition("/")
    count = float(count_s)
    period_s = (period_s or "minute").strip().lower()
    if period_s in _PERIODS:
        period = float(_PERIODS[period_s])
    else:
        period = float(period_s)
    if count <= 0 or period <= 0:
        return None
    return count, period


class TokenBucketLimiter:
    """Thread-safe token buckets with LRU eviction to bound memory."""

    def __init__(self, max_keys: int = 20000):
        self._buckets: "OrderedDict[str, Tuple[float, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def hit(self, key: str, capacity: float, period: float, now: Optional[float] = None) -> Tuple[bool, float]:
        """Consume one token. Returns (allowed, retry_after_seconds)."""
        now = time.monotonic() if now is None else now
        rate = capacity / period
        with self._lock:
            tokens, last = self._buckets.pop(key, (capacity, now))
            tokens = min(capacity, tokens + (now - last) * rate)
            if tokens >= 1.0:
                tokens -= 1.0
                allowed, retry_after = True, 0.0
            else:
                allowed, retry_after = False, (1.0 - tokens) / rate
            self._buckets[key] = (tokens, now)
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
            return allowed, retry_after

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()


limiter = TokenBucketLimiter()


@lru_cache(maxsize=16)
def _parse_trusted(spec: str) -> Tuple[bool, Tuple[ipaddress._BaseNetwork, ...]]:
    trust_any = False
    nets: List[ipaddress._BaseNetwork] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if part == "*":
            trust_any = True
            continue
        try:
            nets.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            logger.warning(f"Ignoring invalid TRUSTED_PROXIES entry: {part!r}")
    return trust_any, tuple(nets)


def _ip_in(addr: str, nets) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    return any(ip in net for net in nets)


def get_client_ip(request: Request) -> str:
    """
    Client IP for rate limiting and hashing.

    X-Forwarded-For is honoured only when the socket peer is a trusted proxy.
    The header is walked right to left, skipping trusted proxies; the first
    untrusted address is the client. Left-most entries (client-controlled)
    are never trusted blindly.
    """
    peer = request.client.host if request.client else "unknown"
    trust_any, nets = _parse_trusted(settings.TRUSTED_PROXIES)
    if not (trust_any or nets):
        return peer
    if not (trust_any or _ip_in(peer, nets)):
        return peer
    xff = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in xff.split(",") if h.strip()]
    for hop in reversed(hops):
        if _ip_in(hop, nets):
            continue
        try:
            return str(ipaddress.ip_address(hop))
        except ValueError:
            return peer
    return peer


def rate_limit(name: str) -> Callable:
    """
    FastAPI dependency factory. ``name`` selects the RATE_LIMIT_<NAME> setting.
    Raises HTTP 429 with Retry-After when the bucket is empty.
    """
    setting_name = f"RATE_LIMIT_{name.upper()}"

    async def _dependency(request: Request) -> None:
        if not settings.RATE_LIMIT_ENABLED:
            return
        rate = parse_rate(getattr(settings, setting_name, None))
        if rate is None:
            return
        capacity, period = rate
        ip = get_client_ip(request)
        allowed, retry_after = limiter.hit(f"{name}:{ip}", capacity, period)
        if not allowed:
            retry = max(1, math.ceil(retry_after))
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "RateLimited",
                    "message": "Too many requests. Please slow down and try again shortly.",
                    "retry_after_seconds": retry,
                },
                headers={"Retry-After": str(retry)},
            )

    _dependency.__name__ = f"rate_limit_{name}"
    return _dependency
