"""
Admin authentication: scrypt password hash, HMAC-signed bearer tokens,
and per-IP login throttling. Standard library only.

Password hash format:  scrypt$<n>$<r>$<p>$<salt_b64>$<hash_b64>
Token format:          <payload_b64url>.<hmac_sha256_b64url>
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from fastapi import HTTPException, Request

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

SCRYPT_N = 2 ** 14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
TOKEN_VERSION = 1

# Used only when ADMIN_TOKEN_SECRET is unset: tokens then stop working on restart.
_PROCESS_SECRET = secrets.token_bytes(32)


def _b64e(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64d(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


# ------------------------------------------------------------------ password

def hash_password(password: str, *, n: int = SCRYPT_N, r: int = SCRYPT_R, p: int = SCRYPT_P) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=SCRYPT_DKLEN)
    return f"scrypt${n}${r}${p}${_b64e(salt)}${_b64e(dk)}"


def verify_password(password: str, encoded: Optional[str]) -> bool:
    if not encoded or not password:
        return False
    try:
        scheme, n_s, r_s, p_s, salt_s, hash_s = encoded.strip().split("$")
        if scheme != "scrypt":
            return False
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt, expected = _b64d(salt_s), _b64d(hash_s)
        dk = hashlib.scrypt(
            password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
            dklen=len(expected), maxmem=256 * 1024 * 1024,
        )
    except (ValueError, TypeError) as e:
        logger.error(f"ADMIN_PASSWORD_HASH is malformed: {type(e).__name__}")
        return False
    return hmac.compare_digest(dk, expected)


def admin_enabled() -> bool:
    return bool(settings.ADMIN_PASSWORD_HASH)


# -------------------------------------------------------------------- tokens

def _signing_key() -> bytes:
    secret = settings.ADMIN_TOKEN_SECRET.encode("utf-8") if settings.ADMIN_TOKEN_SECRET else _PROCESS_SECRET
    # Binding the key to the password hash means rotating the password
    # invalidates every token issued before.
    material = b"admin-token-v1|" + (settings.ADMIN_PASSWORD_HASH or "").encode("utf-8")
    return hmac.new(secret, material, hashlib.sha256).digest()


def _sign(payload_b64: str) -> str:
    return _b64e(hmac.new(_signing_key(), payload_b64.encode("ascii"), hashlib.sha256).digest())


def issue_token(now: Optional[float] = None, ttl: Optional[int] = None) -> Tuple[str, int]:
    now_i = int(now if now is not None else time.time())
    exp = now_i + int(ttl if ttl is not None else settings.ADMIN_TOKEN_TTL_SECONDS)
    payload = {"v": TOKEN_VERSION, "sub": "admin", "iat": now_i, "exp": exp, "jti": secrets.token_hex(8)}
    payload_b64 = _b64e(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    return f"{payload_b64}.{_sign(payload_b64)}", exp


def verify_token(token: str, now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    if not token or len(token) > 2048 or token.count(".") != 1:
        return None
    payload_b64, sig = token.split(".")
    if not hmac.compare_digest(_sign(payload_b64).encode("ascii"), sig.encode("ascii", "replace")):
        return None
    try:
        payload = json.loads(_b64d(payload_b64))
    except (ValueError, TypeError):
        return None
    now_f = time.time() if now is None else now
    if (
        not isinstance(payload, dict)
        or payload.get("v") != TOKEN_VERSION
        or payload.get("sub") != "admin"
        or not isinstance(payload.get("exp"), int)
        or payload["exp"] <= now_f
        or payload.get("iat", 0) > now_f + 60
    ):
        return None
    return payload


async def require_admin(request: Request) -> Dict[str, Any]:
    """Dependency: valid bearer token or 401. Cookies are never consulted."""
    unauthorized = HTTPException(
        status_code=401,
        detail={"error": "Unauthorized", "message": "Admin authentication required"},
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not admin_enabled():
        raise unauthorized
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise unauthorized
    claims = verify_token(token.strip())
    if claims is None:
        raise unauthorized
    return claims


# ------------------------------------------------------------ login throttle

class LoginThrottle:
    """Lock an IP out after too many failed logins within the lockout window."""

    def __init__(self, max_keys: int = 10000):
        self._failures: Dict[str, Deque[float]] = {}
        self._locked_until: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._max_keys = max_keys

    def retry_after(self, ip: str, now: Optional[float] = None) -> int:
        now = time.monotonic() if now is None else now
        with self._lock:
            until = self._locked_until.get(ip, 0.0)
            if until > now:
                return max(1, int(until - now + 0.999))
            self._locked_until.pop(ip, None)
            return 0

    def record_failure(self, ip: str, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        window = settings.ADMIN_LOGIN_LOCKOUT_SECONDS
        with self._lock:
            if len(self._failures) > self._max_keys:
                self._failures.clear()
            q = self._failures.setdefault(ip, deque())
            q.append(now)
            while q and q[0] < now - window:
                q.popleft()
            if len(q) >= settings.ADMIN_LOGIN_MAX_FAILURES:
                self._locked_until[ip] = now + window
                q.clear()
                logger.warning("Admin login locked out for a client after repeated failures")

    def record_success(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)
            self._locked_until.pop(ip, None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()
            self._locked_until.clear()


login_throttle = LoginThrottle()
