"""
Prediction and feedback store.

SQLite database (WAL mode) plus, only when the user explicitly consents,
re-encoded images on disk:

    DATA_DIR/app.db
    DATA_DIR/uploads/YYYY-MM/<prediction_id>.webp

Privacy rules enforced here:
- Prediction metadata and the SHA-256 of the upload are always recorded.
- Image bytes are written only through ``save_consented_image``, which the
  API calls only when ``consent_store_image`` is true.
- Stored images are decoded, orientation-corrected, rebuilt from raw pixels
  (dropping EXIF / XMP / ICC / text chunks) and re-encoded as WebP.
- Raw client IPs are never stored; only a salted HMAC.
- Everything older than RETENTION_DAYS is purged (rows, feedback and files).

All public methods are synchronous and thread-safe; call them from request
handlers through ``asyncio.to_thread`` so the event loop is never blocked.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import io
import json
import os
import secrets
import sqlite3
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_TS_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    stored_image_path TEXT,
    width INTEGER,
    height INTEGER,
    format TEXT,
    final_prob REAL NOT NULL,
    final_pred TEXT NOT NULL,
    fusion_used INTEGER NOT NULL DEFAULT 1,
    submodel_json TEXT,
    explain_json TEXT,
    llm_json TEXT,
    manifest_hash TEXT,
    variant TEXT NOT NULL DEFAULT 'prod',
    variant_json TEXT,
    latency_ms INTEGER,
    client_hash TEXT,
    consent_store_image INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at, id);
CREATE INDEX IF NOT EXISTS idx_predictions_variant ON predictions(variant);
CREATE INDEX IF NOT EXISTS idx_predictions_sha ON predictions(image_sha256);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    prediction_id TEXT NOT NULL UNIQUE REFERENCES predictions(id) ON DELETE CASCADE,
    user_label TEXT NOT NULL CHECK (user_label IN ('ai', 'real', 'unsure')),
    user_knows_truth INTEGER NOT NULL,
    verdict_correct TEXT NOT NULL CHECK (verdict_correct IN ('yes', 'no', 'unsure')),
    source_note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Columns returned by list/export queries (explain_json is deliberately excluded:
# heatmaps are large and only needed in the detail view).
_LIST_COLUMNS = """
    p.id, p.created_at, p.image_sha256, p.stored_image_path, p.width, p.height,
    p.format, p.final_prob, p.final_pred, p.fusion_used, p.submodel_json,
    p.manifest_hash, p.variant, p.latency_ms,
    f.user_label, f.user_knows_truth, f.verdict_correct, f.source_note,
    f.created_at AS feedback_at
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def format_ts(dt: datetime) -> str:
    """Fixed-width UTC timestamp, so lexical order equals time order."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime(_TS_FORMAT)


def derive_verdict_correct(final_pred: str, user_knows_truth: bool, user_label: str) -> str:
    """Compare the model verdict with the user's label: yes / no / unsure."""
    if not user_knows_truth or user_label not in ("ai", "real"):
        return "unsure"
    model_says_ai = final_pred == "fake"
    user_says_ai = user_label == "ai"
    return "yes" if model_says_ai == user_says_ai else "no"


def sanitize_image_for_storage(image_bytes: bytes, max_side: int = 2048, quality: int = 90) -> bytes:
    """
    Decode an upload and re-encode it as metadata-free WebP.

    The pixels are copied into a brand-new Image so that no EXIF, XMP, ICC
    profile, PNG text chunk or other ancillary data survives. EXIF orientation
    is applied first so the stored image still looks upright.
    """
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(image_bytes)) as src:
        src.seek(0)  # first frame of animated formats
        img = ImageOps.exif_transpose(src)
        has_alpha = "A" in img.getbands() or "transparency" in img.info
        mode = "RGBA" if has_alpha else "RGB"
        if img.mode != mode:
            img = img.convert(mode)
        clean = Image.frombytes(mode, img.size, img.tobytes())

    if max_side > 0 and max(clean.size) > max_side:
        clean.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)

    out = io.BytesIO()
    clean.save(out, format="WEBP", quality=quality, method=4)
    return out.getvalue()


def _probe_image(image_bytes: bytes) -> Tuple[Optional[int], Optional[int], Optional[str]]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.width, img.height, img.format
    except Exception:
        return None, None, None


def resolve_data_dir(configured: Optional[str] = None) -> Path:
    """
    Pick the data directory.

    An explicit DATA_DIR wins. Otherwise use the HF Spaces persistent volume
    (/data) when it exists and is writable, then ./.local_data, then a temp dir.
    """
    candidates: List[Tuple[Path, bool]] = []
    if configured:
        candidates.append((Path(configured), True))
    else:
        candidates.append((Path("/data"), False))
        candidates.append((Path(".local_data"), True))
        candidates.append((Path(tempfile.gettempdir()) / "deepfake-detector-data", True))

    for path, may_create in candidates:
        try:
            if not path.is_dir():
                if not may_create:
                    continue
                path.mkdir(parents=True, exist_ok=True)
            probe = path / f".write-test-{secrets.token_hex(4)}"
            probe.write_bytes(b"")
            probe.unlink()
            return path.resolve()
        except OSError:
            continue
    raise RuntimeError("No writable data directory available for the prediction store")


class Store:
    """Thread-safe SQLite store for predictions, feedback and consented images."""

    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir).resolve()
        self.uploads_dir = self.data_dir / "uploads"
        self.db_path = self.data_dir / "app.db"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._conns: List[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._closed = False

        conn = self._conn()
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(_SCHEMA)
        self._client_salt = self._load_client_salt()
        logger.info(f"Prediction store ready at {self.db_path}")

    # ------------------------------------------------------------------ infra

    def _conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("Store is closed")
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
            with self._conns_lock:
                self._conns.append(conn)
        return conn

    def close(self) -> None:
        self._closed = True
        with self._conns_lock:
            for conn in self._conns:
                try:
                    conn.close()
                except Exception:
                    pass
            self._conns.clear()

    def _write(self, fn):
        """Run fn(conn) inside a serialized write transaction."""
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN IMMEDIATE")
            try:
                result = fn(conn)
                conn.execute("COMMIT")
                return result
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def _load_client_salt(self) -> bytes:
        if settings.CLIENT_HASH_SALT:
            return settings.CLIENT_HASH_SALT.encode("utf-8")

        def _get_or_create(conn):
            row = conn.execute("SELECT value FROM meta WHERE key='client_hash_salt'").fetchone()
            if row:
                return row["value"]
            value = secrets.token_hex(32)
            conn.execute("INSERT INTO meta(key, value) VALUES('client_hash_salt', ?)", (value,))
            return value

        return self._write(_get_or_create).encode("utf-8")

    def client_hash(self, client_ip: Optional[str]) -> Optional[str]:
        if not client_ip:
            return None
        return hmac.new(self._client_salt, client_ip.encode("utf-8"), hashlib.sha256).hexdigest()[:32]

    def _abs_upload_path(self, rel_path: str) -> Optional[Path]:
        """Resolve a stored relative path, refusing anything outside uploads/."""
        try:
            path = (self.data_dir / rel_path).resolve()
            path.relative_to(self.uploads_dir)
            return path
        except (ValueError, OSError):
            return None

    # ------------------------------------------------------------ predictions

    def record_prediction(
        self,
        *,
        image_bytes: bytes,
        final_pred: str,
        final_prob: float,
        fusion_used: bool,
        submodels: Optional[Dict[str, Any]] = None,
        explain: Optional[Dict[str, Any]] = None,
        variant: str = "prod",
        variant_info: Optional[Dict[str, Any]] = None,
        latency_ms: Optional[int] = None,
        client_ip: Optional[str] = None,
        manifest_hash: Optional[str] = None,
        consent_store_image: bool = False,
        prediction_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
    ) -> str:
        """Insert prediction metadata (never image bytes). Returns the prediction id."""
        pid = prediction_id or uuid.uuid4().hex
        sha = hashlib.sha256(image_bytes).hexdigest()
        width, height, fmt = _probe_image(image_bytes)
        row = (
            pid,
            format_ts(created_at or utcnow()),
            sha,
            width,
            height,
            fmt,
            float(final_prob),
            final_pred,
            1 if fusion_used else 0,
            json.dumps(submodels or {}),
            json.dumps(explain or {}),
            manifest_hash,
            variant or "prod",
            json.dumps(variant_info) if variant_info else None,
            latency_ms,
            self.client_hash(client_ip),
            1 if consent_store_image else 0,
        )
        self._write(lambda conn: conn.execute(
            """
            INSERT INTO predictions (
                id, created_at, image_sha256, width, height, format, final_prob,
                final_pred, fusion_used, submodel_json, explain_json, manifest_hash,
                variant, variant_json, latency_ms, client_hash, consent_store_image
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        ))
        return pid

    def save_consented_image(self, prediction_id: str, image_bytes: bytes) -> Optional[str]:
        """
        Re-encode and store an image for a prediction whose user consented.
        Never raises; returns the relative path or None.
        """
        try:
            row = self._conn().execute(
                "SELECT created_at, consent_store_image FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()
            if row is None or not row["consent_store_image"]:
                return None

            webp = sanitize_image_for_storage(
                image_bytes,
                max_side=settings.STORED_IMAGE_MAX_SIDE,
                quality=settings.STORED_IMAGE_WEBP_QUALITY,
            )
            month = row["created_at"][:7]
            rel = f"uploads/{month}/{prediction_id}.webp"
            dest = self._abs_upload_path(rel)
            if dest is None:
                return None
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".webp.tmp")
            tmp.write_bytes(webp)
            os.replace(tmp, dest)

            updated = self._write(lambda conn: conn.execute(
                "UPDATE predictions SET stored_image_path = ? WHERE id = ?",
                (rel, prediction_id),
            ).rowcount)
            if not updated:
                # Row vanished (deleted or purged) while we were encoding.
                dest.unlink(missing_ok=True)
                return None
            return rel
        except Exception as e:
            logger.warning(f"Failed to store consented image for {prediction_id}: {e}")
            return None

    def add_llm_insight(self, prediction_id: str, model_name: str, insight: Dict[str, Any]) -> bool:
        def _merge(conn):
            row = conn.execute(
                "SELECT llm_json FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            if row is None:
                return False
            data = json.loads(row["llm_json"] or "{}")
            data[model_name] = insight
            conn.execute(
                "UPDATE predictions SET llm_json = ? WHERE id = ?",
                (json.dumps(data), prediction_id),
            )
            return True

        return self._write(_merge)

    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        row = self._conn().execute(
            """
            SELECT p.*, f.user_label, f.user_knows_truth, f.verdict_correct,
                   f.source_note, f.created_at AS feedback_at
            FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id
            WHERE p.id = ?
            """,
            (prediction_id,),
        ).fetchone()
        if row is None:
            return None
        item = self._row_to_item(row)
        item["explain"] = json.loads(row["explain_json"] or "{}")
        item["llm"] = json.loads(row["llm_json"] or "{}")
        item["variant_info"] = json.loads(row["variant_json"]) if row["variant_json"] else None
        return item

    def get_image_bytes(self, prediction_id: str, thumb_size: Optional[int] = None) -> Optional[bytes]:
        row = self._conn().execute(
            "SELECT stored_image_path FROM predictions WHERE id = ?", (prediction_id,)
        ).fetchone()
        if row is None or not row["stored_image_path"]:
            return None
        path = self._abs_upload_path(row["stored_image_path"])
        if path is None or not path.is_file():
            return None
        data = path.read_bytes()
        if thumb_size:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as img:
                img.thumbnail((thumb_size, thumb_size))
                out = io.BytesIO()
                img.save(out, format="WEBP", quality=75)
                data = out.getvalue()
        return data

    def delete_prediction(self, prediction_id: str) -> bool:
        def _delete(conn):
            row = conn.execute(
                "SELECT stored_image_path FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            if row is None:
                return None
            conn.execute("DELETE FROM feedback WHERE prediction_id = ?", (prediction_id,))
            conn.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            return row["stored_image_path"] or ""

        rel = self._write(_delete)
        if rel is None:
            return False
        if rel:
            path = self._abs_upload_path(rel)
            if path is not None:
                path.unlink(missing_ok=True)
        return True

    # --------------------------------------------------------------- feedback

    def upsert_feedback(
        self,
        *,
        prediction_id: str,
        user_knows_truth: bool,
        user_label: str,
        source_note: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Create or replace the single feedback entry of a prediction. None if unknown id."""
        if not user_knows_truth:
            user_label = "unsure"

        def _upsert(conn):
            row = conn.execute(
                "SELECT final_pred FROM predictions WHERE id = ?", (prediction_id,)
            ).fetchone()
            if row is None:
                return None
            verdict = derive_verdict_correct(row["final_pred"], user_knows_truth, user_label)
            now = format_ts(utcnow())
            conn.execute(
                """
                INSERT INTO feedback (
                    id, prediction_id, user_label, user_knows_truth, verdict_correct,
                    source_note, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(prediction_id) DO UPDATE SET
                    user_label = excluded.user_label,
                    user_knows_truth = excluded.user_knows_truth,
                    verdict_correct = excluded.verdict_correct,
                    source_note = excluded.source_note,
                    updated_at = excluded.updated_at
                """,
                (
                    uuid.uuid4().hex, prediction_id, user_label,
                    1 if user_knows_truth else 0, verdict, source_note, now, now,
                ),
            )
            return {
                "prediction_id": prediction_id,
                "user_knows_truth": bool(user_knows_truth),
                "user_label": user_label,
                "verdict_correct": verdict,
            }

        return self._write(_upsert)

    # ------------------------------------------------------------------ admin

    @staticmethod
    def _row_to_item(row: sqlite3.Row) -> Dict[str, Any]:
        keys = row.keys()
        feedback = None
        if row["user_label"] is not None:
            feedback = {
                "user_label": row["user_label"],
                "user_knows_truth": bool(row["user_knows_truth"]),
                "verdict_correct": row["verdict_correct"],
                "source_note": row["source_note"],
                "created_at": row["feedback_at"],
            }
        return {
            "id": row["id"],
            "created_at": row["created_at"],
            "image_sha256": row["image_sha256"],
            "has_image": bool(row["stored_image_path"]),
            "width": row["width"],
            "height": row["height"],
            "format": row["format"],
            "final_prob": row["final_prob"],
            "final_pred": row["final_pred"],
            "fusion_used": bool(row["fusion_used"]),
            "submodels": json.loads(row["submodel_json"] or "{}"),
            "manifest_hash": row["manifest_hash"],
            "variant": row["variant"],
            "latency_ms": row["latency_ms"] if "latency_ms" in keys else None,
            "feedback": feedback,
        }

    def _filters_sql(
        self,
        *,
        disagreement: Optional[bool] = None,
        uncertain: Optional[bool] = None,
        has_feedback: Optional[bool] = None,
        labelled: Optional[bool] = None,
        has_image: Optional[bool] = None,
        variant: Optional[str] = None,
        created_from: Optional[datetime] = None,
        created_before: Optional[datetime] = None,
    ) -> Tuple[List[str], List[Any]]:
        where: List[str] = []
        args: List[Any] = []
        if disagreement is True:
            where.append("f.verdict_correct = 'no'")
        elif disagreement is False:
            where.append("(f.verdict_correct IS NULL OR f.verdict_correct != 'no')")
        if uncertain is not None:
            clause = "p.final_prob BETWEEN ? AND ?"
            where.append(clause if uncertain else f"NOT ({clause})")
            args += [settings.UNCERTAIN_LOW, settings.UNCERTAIN_HIGH]
        if has_feedback is True:
            where.append("f.id IS NOT NULL")
        elif has_feedback is False:
            where.append("f.id IS NULL")
        if labelled is True:
            where.append("f.verdict_correct IN ('yes', 'no')")
        elif labelled is False:
            where.append("(f.verdict_correct IS NULL OR f.verdict_correct = 'unsure')")
        if has_image is True:
            where.append("p.stored_image_path IS NOT NULL")
        elif has_image is False:
            where.append("p.stored_image_path IS NULL")
        if variant:
            where.append("p.variant = ?")
            args.append(variant)
        if created_from is not None:
            where.append("p.created_at >= ?")
            args.append(format_ts(created_from))
        if created_before is not None:
            where.append("p.created_at < ?")
            args.append(format_ts(created_before))
        return where, args

    def list_predictions(
        self,
        *,
        limit: int = 50,
        cursor: Optional[Tuple[str, str]] = None,
        **filters: Any,
    ) -> Tuple[List[Dict[str, Any]], Optional[Tuple[str, str]]]:
        """Newest first, keyset-paginated on (created_at, id)."""
        limit = max(1, min(int(limit), 200))
        where, args = self._filters_sql(**filters)
        if cursor is not None:
            where.append("(p.created_at < ? OR (p.created_at = ? AND p.id < ?))")
            args += [cursor[0], cursor[0], cursor[1]]
        sql = f"SELECT {_LIST_COLUMNS} FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY p.created_at DESC, p.id DESC LIMIT ?"
        rows = self._conn().execute(sql, args + [limit + 1]).fetchall()
        items = [self._row_to_item(r) for r in rows[:limit]]
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = (last["created_at"], last["id"])
        return items, next_cursor

    def iter_export_rows(self, **filters: Any) -> Iterable[Dict[str, Any]]:
        where, args = self._filters_sql(**filters)
        sql = (
            f"SELECT {_LIST_COLUMNS}, p.stored_image_path AS image_path "
            "FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id"
        )
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY p.created_at DESC, p.id DESC"
        for row in self._conn().execute(sql, args):
            item = self._row_to_item(row)
            item["stored_image_path"] = row["image_path"]
            yield item

    def stats(self, days: int = 30, now: Optional[datetime] = None) -> Dict[str, Any]:
        now = now or utcnow()
        since = format_ts(now - timedelta(days=days))
        low, high = settings.UNCERTAIN_LOW, settings.UNCERTAIN_HIGH
        conn = self._conn()
        base = "FROM predictions p LEFT JOIN feedback f ON f.prediction_id = p.id WHERE p.created_at >= ?"

        t = conn.execute(
            f"""
            SELECT COUNT(*) AS predictions,
                   COALESCE(SUM(p.stored_image_path IS NOT NULL), 0) AS with_image,
                   COALESCE(SUM(p.final_pred = 'fake'), 0) AS predicted_fake,
                   COALESCE(SUM(p.final_prob BETWEEN ? AND ?), 0) AS uncertain,
                   COUNT(f.id) AS feedback,
                   COALESCE(SUM(f.verdict_correct IN ('yes', 'no')), 0) AS labelled,
                   COALESCE(SUM(f.verdict_correct = 'yes'), 0) AS agreed
            {base}
            """,
            (low, high, since),
        ).fetchone()
        totals = dict(t)
        totals["agreement_rate"] = (totals["agreed"] / totals["labelled"]) if totals["labelled"] else None

        per_day = [
            {
                **dict(r),
                "agreement_rate": (r["agreed"] / r["labelled"]) if r["labelled"] else None,
            }
            for r in conn.execute(
                f"""
                SELECT substr(p.created_at, 1, 10) AS date,
                       COUNT(*) AS predictions,
                       COUNT(f.id) AS feedback,
                       COALESCE(SUM(f.verdict_correct IN ('yes', 'no')), 0) AS labelled,
                       COALESCE(SUM(f.verdict_correct = 'yes'), 0) AS agreed
                {base}
                GROUP BY date ORDER BY date
                """,
                (since,),
            )
        ]

        confusion = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for r in conn.execute(
            f"""
            SELECT p.final_pred AS pred, f.user_label AS label, COUNT(*) AS n
            {base} AND f.verdict_correct IN ('yes', 'no')
            GROUP BY pred, label
            """,
            (since,),
        ):
            key = {
                ("fake", "ai"): "tp",
                ("fake", "real"): "fp",
                ("real", "real"): "tn",
                ("real", "ai"): "fn",
            }.get((r["pred"], r["label"]))
            if key:
                confusion[key] += r["n"]

        per_variant = [
            {
                **dict(r),
                "agreement_rate": (r["agreed"] / r["labelled"]) if r["labelled"] else None,
                "uncertain_rate": (r["uncertain"] / r["predictions"]) if r["predictions"] else None,
            }
            for r in conn.execute(
                f"""
                SELECT p.variant AS variant,
                       COUNT(*) AS predictions,
                       COALESCE(SUM(p.final_prob BETWEEN ? AND ?), 0) AS uncertain,
                       AVG(p.final_prob) AS mean_prob_fake,
                       COUNT(f.id) AS feedback,
                       COALESCE(SUM(f.verdict_correct IN ('yes', 'no')), 0) AS labelled,
                       COALESCE(SUM(f.verdict_correct = 'yes'), 0) AS agreed
                {base}
                GROUP BY p.variant ORDER BY predictions DESC
                """,
                (low, high, since),
            )
        ]

        calibration = [
            dict(r)
            for r in conn.execute(
                f"""
                SELECT MIN(CAST(p.final_prob * 5 AS INTEGER), 4) AS bin,
                       COUNT(*) AS n,
                       AVG(p.final_prob) AS mean_prob_fake,
                       AVG(f.user_label = 'ai') AS frac_ai
                {base} AND f.verdict_correct IN ('yes', 'no')
                GROUP BY bin ORDER BY bin
                """,
                (since,),
            )
        ]

        return {
            "days": days,
            "since": since,
            "uncertain_band": [low, high],
            "totals": totals,
            "per_day": per_day,
            "confusion": confusion,
            "per_variant": per_variant,
            "calibration": calibration,
        }

    # ------------------------------------------------------------- retention

    def purge(self, retention_days: Optional[int] = None, now: Optional[datetime] = None) -> Dict[str, int]:
        """Delete predictions, feedback and images older than the retention window."""
        days = settings.RETENTION_DAYS if retention_days is None else retention_days
        if days <= 0:
            return {"predictions": 0, "images": 0}
        cutoff = format_ts((now or utcnow()) - timedelta(days=days))

        def _purge(conn):
            rows = conn.execute(
                "SELECT id, stored_image_path FROM predictions WHERE created_at < ?", (cutoff,)
            ).fetchall()
            conn.execute(
                "DELETE FROM feedback WHERE prediction_id IN "
                "(SELECT id FROM predictions WHERE created_at < ?)",
                (cutoff,),
            )
            conn.execute("DELETE FROM predictions WHERE created_at < ?", (cutoff,))
            return [r["stored_image_path"] for r in rows], len(rows)

        paths, n_rows = self._write(_purge)
        n_images = 0
        for rel in paths:
            if not rel:
                continue
            path = self._abs_upload_path(rel)
            if path is not None and path.is_file():
                path.unlink(missing_ok=True)
                n_images += 1
        n_images += self._sweep_orphans()
        if n_rows or n_images:
            logger.info(f"Retention purge removed {n_rows} predictions and {n_images} images")
        return {"predictions": n_rows, "images": n_images}

    def _sweep_orphans(self) -> int:
        """Remove upload files with no matching row (crash leftovers, temp files)."""
        removed = 0
        conn = self._conn()
        # Skip very recent files: a consented image may be mid-write.
        min_age_cutoff = utcnow().timestamp() - 3600
        for month_dir in self.uploads_dir.iterdir():
            if not month_dir.is_dir():
                continue
            for f in month_dir.iterdir():
                try:
                    if f.stat().st_mtime > min_age_cutoff:
                        continue
                except OSError:
                    continue
                stem = f.name.split(".", 1)[0]
                if f.name.endswith(".tmp"):
                    keep = False
                else:
                    keep = conn.execute(
                        "SELECT 1 FROM predictions WHERE id = ? AND stored_image_path IS NOT NULL",
                        (stem,),
                    ).fetchone() is not None
                if not keep:
                    try:
                        f.unlink()
                        removed += 1
                    except OSError:
                        pass
            try:
                month_dir.rmdir()  # only succeeds when empty
            except OSError:
                pass
        return removed


# ---------------------------------------------------------------- singleton

_store: Optional[Store] = None
_store_failed = False
_store_lock = threading.Lock()


def get_store() -> Optional[Store]:
    """Return the process-wide store, or None when disabled or unavailable."""
    global _store, _store_failed
    if not settings.STORE_ENABLED:
        return None
    if _store is not None or _store_failed:
        return _store
    with _store_lock:
        if _store is None and not _store_failed:
            try:
                _store = Store(resolve_data_dir(settings.DATA_DIR))
            except Exception as e:
                _store_failed = True
                logger.error(f"Prediction store unavailable, continuing without it: {e}")
    return _store


def set_store(store: Optional[Store]) -> None:
    """Replace the process-wide store (used by tests)."""
    global _store, _store_failed
    with _store_lock:
        _store = store
        _store_failed = False


async def retention_loop() -> None:
    """Background task: purge expired data at startup and then periodically."""
    interval = max(settings.PURGE_INTERVAL_HOURS, 0.1) * 3600
    while True:
        try:
            store = await asyncio.to_thread(get_store)
            if store is not None:
                await asyncio.to_thread(store.purge)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"Retention purge failed: {e}")
        await asyncio.sleep(interval)
