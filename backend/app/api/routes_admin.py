"""
Admin dashboard API (/admin/*).

Every endpoint except /admin/login requires an HMAC-signed bearer token.
Admin is disabled (all calls 401, login 503) unless ADMIN_PASSWORD_HASH is set.
"""

import asyncio
import base64
import csv
import io
import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from app.core.logging import get_logger
from app.schemas.feedback import AdminLoginRequest, AdminLoginResponse
from app.core.config import settings
from app.services.admin_auth import (
    admin_enabled,
    issue_token,
    login_throttle,
    require_admin,
    verify_password,
)
from app.services.rate_limit import get_client_ip, rate_limit
from app.services.store import Store, get_store

logger = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])
protected = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])

NO_STORE_HEADERS = {
    "Cache-Control": "no-store",
    "X-Robots-Tag": "noindex, nofollow",
}


async def admin_headers_middleware(request: Request, call_next):
    """Admin responses must never be cached or indexed."""
    response = await call_next(request)
    if request.url.path.startswith("/admin"):
        for k, v in NO_STORE_HEADERS.items():
            response.headers[k] = v
    return response


async def _store() -> Store:
    store = await asyncio.to_thread(get_store)
    if store is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "StoreUnavailable", "message": "Prediction store is unavailable"},
        )
    return store


def _encode_cursor(cursor: Optional[Tuple[str, str]]) -> Optional[str]:
    if cursor is None:
        return None
    return base64.urlsafe_b64encode(json.dumps(list(cursor)).encode()).decode().rstrip("=")


def _decode_cursor(cursor: Optional[str]) -> Optional[Tuple[str, str]]:
    if not cursor:
        return None
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4))
        created_at, pid = json.loads(raw)
        if not isinstance(created_at, str) or not isinstance(pid, str):
            raise ValueError
        return created_at, pid
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "BadCursor", "message": "Invalid cursor"})


def _day_start(d: date) -> datetime:
    return datetime.combine(d, time.min, tzinfo=timezone.utc)


def _filters(
    disagreement: Optional[bool] = Query(None, description="Only items where the user label contradicts the verdict"),
    uncertain: Optional[bool] = Query(None, description="Only items with final probability in the uncertain band"),
    has_feedback: Optional[bool] = Query(None),
    labelled: Optional[bool] = Query(None, description="Only items with a known user label (ai/real)"),
    has_image: Optional[bool] = Query(None),
    variant: Optional[str] = Query(None, max_length=64),
    date_from: Optional[date] = Query(None, description="Inclusive, UTC (YYYY-MM-DD)"),
    date_to: Optional[date] = Query(None, description="Inclusive, UTC (YYYY-MM-DD)"),
) -> Dict[str, Any]:
    return {
        "disagreement": disagreement,
        "uncertain": uncertain,
        "has_feedback": has_feedback,
        "labelled": labelled,
        "has_image": has_image,
        "variant": variant,
        "created_from": _day_start(date_from) if date_from else None,
        "created_before": _day_start(date_to) + timedelta(days=1) if date_to else None,
    }


# --------------------------------------------------------------------- login

@router.post(
    "/login",
    response_model=AdminLoginResponse,
    dependencies=[Depends(rate_limit("admin_login"))],
    responses={401: {"description": "Wrong password"}, 429: {"description": "Too many attempts"},
               503: {"description": "Admin disabled"}},
)
async def admin_login(body: AdminLoginRequest, request: Request) -> AdminLoginResponse:
    if not admin_enabled():
        raise HTTPException(
            status_code=503,
            detail={"error": "AdminDisabled", "message": "Admin dashboard is not configured"},
        )
    ip = get_client_ip(request)
    retry = login_throttle.retry_after(ip)
    if retry:
        raise HTTPException(
            status_code=429,
            detail={"error": "TooManyAttempts", "message": "Too many failed logins. Try again later.",
                    "retry_after_seconds": retry},
            headers={"Retry-After": str(retry)},
        )
    ok = await asyncio.to_thread(verify_password, body.password, settings.ADMIN_PASSWORD_HASH)
    if not ok:
        login_throttle.record_failure(ip)
        raise HTTPException(
            status_code=401,
            detail={"error": "Unauthorized", "message": "Invalid password"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    login_throttle.record_success(ip)
    token, exp = issue_token()
    logger.info("Admin login succeeded")
    return AdminLoginResponse(access_token=token, expires_at=exp)


# ----------------------------------------------------------------- protected

@protected.get("/session")
async def admin_session(claims: Dict[str, Any] = Depends(require_admin)) -> Dict[str, Any]:
    """Check that the current token is valid."""
    return {"ok": True, "expires_at": claims["exp"], "retention_days": settings.RETENTION_DAYS}


@protected.get("/predictions")
async def list_predictions(
    limit: int = Query(50, ge=1, le=200),
    cursor: Optional[str] = Query(None, max_length=512),
    filters: Dict[str, Any] = Depends(_filters),
) -> Dict[str, Any]:
    store = await _store()
    items, next_cursor = await asyncio.to_thread(
        store.list_predictions, limit=limit, cursor=_decode_cursor(cursor), **filters
    )
    return {"items": items, "next_cursor": _encode_cursor(next_cursor)}


@protected.get("/predictions/{prediction_id}")
async def get_prediction(prediction_id: str) -> Dict[str, Any]:
    store = await _store()
    item = await asyncio.to_thread(store.get_prediction, prediction_id)
    if item is None:
        raise HTTPException(status_code=404, detail={"error": "NotFound", "message": "Unknown prediction"})
    return item


@protected.get("/predictions/{prediction_id}/image")
async def get_prediction_image(
    prediction_id: str,
    thumb: Optional[int] = Query(None, ge=32, le=512, description="Return a thumbnail with this max side"),
) -> Response:
    store = await _store()
    data = await asyncio.to_thread(store.get_image_bytes, prediction_id, thumb)
    if data is None:
        raise HTTPException(status_code=404, detail={"error": "NotFound", "message": "No stored image"})
    return Response(content=data, media_type="image/webp", headers=NO_STORE_HEADERS)


@protected.delete("/predictions/{prediction_id}", status_code=204)
async def delete_prediction(prediction_id: str) -> Response:
    store = await _store()
    deleted = await asyncio.to_thread(store.delete_prediction, prediction_id)
    if not deleted:
        raise HTTPException(status_code=404, detail={"error": "NotFound", "message": "Unknown prediction"})
    logger.info("Admin deleted a prediction record")
    return Response(status_code=204)


@protected.get("/stats")
async def get_stats(days: int = Query(30, ge=1, le=3650)) -> Dict[str, Any]:
    store = await _store()
    return await asyncio.to_thread(store.stats, days)


_CSV_COLUMNS = [
    "id", "created_at", "variant", "manifest_hash", "final_pred", "final_prob",
    "submodels_json", "user_knows_truth", "user_label", "verdict_correct", "source_note",
    "image_sha256", "has_image", "stored_image_path", "width", "height", "format", "latency_ms",
]


def _csv_safe(value: Any) -> Any:
    """Neutralise spreadsheet formula injection in free-text cells."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _build_csv(rows) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_COLUMNS)
    for item in rows:
        fb = item.get("feedback") or {}
        writer.writerow([_csv_safe(v) for v in (
            item["id"], item["created_at"], item["variant"], item["manifest_hash"],
            item["final_pred"], item["final_prob"], json.dumps(item["submodels"]),
            fb.get("user_knows_truth"), fb.get("user_label"), fb.get("verdict_correct"),
            fb.get("source_note"), item["image_sha256"], item["has_image"],
            item.get("stored_image_path"), item["width"], item["height"], item["format"],
            item["latency_ms"],
        )])
    return buf.getvalue()


@protected.get("/export.csv")
async def export_csv(filters: Dict[str, Any] = Depends(_filters)) -> Response:
    store = await _store()
    content = await asyncio.to_thread(lambda: _build_csv(store.iter_export_rows(**filters)))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={**NO_STORE_HEADERS, "Content-Disposition": f'attachment; filename="predictions-{stamp}.csv"'},
    )


@protected.post("/purge")
async def run_purge() -> Dict[str, Any]:
    """Apply the retention policy now."""
    store = await _store()
    removed = await asyncio.to_thread(store.purge)
    return {"retention_days": settings.RETENTION_DAYS, "removed": removed}

