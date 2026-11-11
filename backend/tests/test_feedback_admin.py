"""
Tests for the prediction store, /feedback, /admin/* and rate limiting.

Inference and fusion are mocked; no models are downloaded.
"""

import base64
import io
import json
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.requests import Request

with patch('app.services.model_registry.ModelRegistry.load_from_fusion_repo', new_callable=AsyncMock):
    from app.main import app

from app.api import routes_predict
from app.core.config import settings
from app.services import admin_auth, rate_limit, store as store_mod
from app.services.admin_auth import hash_password, issue_token, verify_password, verify_token
from app.services.store import Store, sanitize_image_for_storage, utcnow

PASSWORD = "correct horse battery staple"
HEATMAP = base64.b64encode(b"\x89PNG fake heatmap").decode()


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "data")
    store_mod.set_store(s)
    yield s
    store_mod.set_store(None)
    s.close()


@pytest.fixture(autouse=True)
def _isolation(monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_PASSWORD_HASH", hash_password(PASSWORD, n=2 ** 12))
    monkeypatch.setattr(settings, "ADMIN_TOKEN_SECRET", "test-token-secret")
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", "")
    monkeypatch.setattr(settings, "STORE_ENABLED", True)
    rate_limit.limiter.reset()
    admin_auth.login_throttle.reset()
    yield
    rate_limit.limiter.reset()
    admin_auth.login_throttle.reset()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_models():
    """Two submodels; fused verdict is 'fake' with p=0.8."""
    inference = MagicMock()
    inference.predict_all_submodels.return_value = {
        "cnn-transfer": {"pred": "fake", "pred_int": 1, "prob_fake": 0.9,
                         "heatmap_base64": HEATMAP, "explainability_type": "grad_cam",
                         "focus_summary": "centre"},
        "vit-base": {"pred": "fake", "pred_int": 1, "prob_fake": 0.7,
                     "heatmap_base64": HEATMAP, "explainability_type": "attention_rollout"},
    }
    fusion = MagicMock()
    fusion.fuse.return_value = {
        "pred": "fake", "pred_int": 1, "prob_fake": 0.8,
        "meta": {"submodel_weights": {"cnn-transfer": 1.0, "vit-base": 0.5},
                 "weighted_contributions": {"cnn-transfer": 0.9, "vit-base": 0.35},
                 "contribution_percentages": {"cnn-transfer": 72.0, "vit-base": 28.0}},
    }
    preprocess = MagicMock()
    with patch.object(routes_predict, "get_inference_service", return_value=inference), \
         patch.object(routes_predict, "get_fusion_service", return_value=fusion), \
         patch.object(routes_predict, "get_preprocess_service", return_value=preprocess):
        yield inference, fusion


def jpeg_with_exif(size=(64, 48)) -> bytes:
    img = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    exif[0x010F] = "SecretCameraMaker"   # Make
    exif[0x0110] = "SecretModel-X1"      # Model
    exif[0x0131] = "SecretSoftware"      # Software
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes(), quality=90)
    data = buf.getvalue()
    assert b"SecretCameraMaker" in data
    return data


def do_predict(client, consent=None, image_bytes=None, headers=None):
    files = {"image": ("photo.jpg", io.BytesIO(image_bytes or jpeg_with_exif()), "image/jpeg")}
    data = {} if consent is None else {"consent_store_image": "true" if consent else "false"}
    return client.post("/predict", files=files, data=data, headers=headers or {})


def login(client) -> str:
    r = client.post("/admin/login", json={"password": PASSWORD})
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def all_files(root: Path):
    return [p for p in root.rglob("*") if p.is_file()]


# ---------------------------------------------------------------- roundtrip

def test_predict_feedback_admin_roundtrip(client, store, mock_models):
    r = do_predict(client)
    assert r.status_code == 200, r.text
    body = r.json()
    pid = body["prediction_id"]
    assert pid and body["final"]["pred"] == "fake"

    r = client.post("/feedback", json={
        "prediction_id": pid, "user_knows_truth": True, "user_label": "ai",
        "source_note": "Made with Midjourney v6",
    })
    assert r.status_code == 200, r.text
    assert r.json()["verdict_correct"] == "yes"

    token = login(client)

    r = client.get("/admin/predictions", headers=auth(token))
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-store"
    items = r.json()["items"]
    assert [i["id"] for i in items] == [pid]
    item = items[0]
    assert item["final_prob"] == pytest.approx(0.8)
    assert item["variant"] == "prod"
    assert item["has_image"] is False
    assert item["submodels"]["cnn-transfer"]["prob_fake"] == pytest.approx(0.9)
    assert item["feedback"]["user_label"] == "ai"
    assert item["feedback"]["source_note"] == "Made with Midjourney v6"

    r = client.get(f"/admin/predictions/{pid}", headers=auth(token))
    assert r.status_code == 200
    detail = r.json()
    # No consent given: heatmaps are not stored, other explain metadata is.
    assert "heatmap_base64" not in detail["explain"]["cnn-transfer"]
    assert detail["explain"]["vit-base"]["explainability_type"] == "attention_rollout"
    assert len(detail["image_sha256"]) == 64

    r = client.get("/admin/stats", headers=auth(token))
    assert r.status_code == 200
    stats = r.json()
    assert stats["totals"]["predictions"] == 1
    assert stats["totals"]["labelled"] == 1
    assert stats["totals"]["agreement_rate"] == 1.0
    assert stats["confusion"] == {"tp": 1, "fp": 0, "tn": 0, "fn": 0}
    assert stats["per_variant"][0]["variant"] == "prod"
    assert len(stats["per_day"]) == 1

    r = client.get("/admin/export.csv?labelled=true", headers=auth(token))
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert pid in r.text and "Midjourney" in r.text


def test_feedback_upsert_and_disagreement(client, store, mock_models):
    pid = do_predict(client).json()["prediction_id"]
    r = client.post("/feedback", json={"prediction_id": pid, "user_knows_truth": True, "user_label": "real"})
    assert r.json()["verdict_correct"] == "no"
    # Second submission replaces the first (one entry per prediction)
    r = client.post("/feedback", json={"prediction_id": pid, "user_knows_truth": False, "user_label": "ai"})
    assert r.status_code == 200
    assert r.json() == {"prediction_id": pid, "user_knows_truth": False,
                        "user_label": "unsure", "verdict_correct": "unsure"}
    count = store._conn().execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    assert count == 1

    client.post("/feedback", json={"prediction_id": pid, "user_knows_truth": True, "user_label": "real"})
    token = login(client)
    r = client.get("/admin/predictions?disagreement=true", headers=auth(token))
    assert [i["id"] for i in r.json()["items"]] == [pid]
    r = client.get("/admin/predictions?disagreement=false", headers=auth(token))
    assert r.json()["items"] == []
    stats = client.get("/admin/stats", headers=auth(token)).json()
    assert stats["confusion"]["fp"] == 1


def test_feedback_validation(client, store, mock_models):
    pid = do_predict(client).json()["prediction_id"]
    bad = [
        {"prediction_id": pid, "user_knows_truth": True, "user_label": "maybe"},
        {"prediction_id": pid, "user_knows_truth": True, "user_label": "ai", "source_note": "x" * 501},
        {"prediction_id": pid, "user_knows_truth": True, "user_label": "ai", "extra": 1},
        {"prediction_id": "../../etc", "user_knows_truth": True, "user_label": "ai"},
    ]
    for payload in bad:
        assert client.post("/feedback", json=payload).status_code == 422, payload
    r = client.post("/feedback", json={"prediction_id": "0" * 32, "user_knows_truth": True, "user_label": "ai"})
    assert r.status_code == 404


# ------------------------------------------------------------- consent/privacy

def test_heatmaps_stored_only_with_consent(client, store, mock_models):
    pid_no = do_predict(client, consent=False).json()["prediction_id"]
    pid_yes = do_predict(client, consent=True).json()["prediction_id"]
    token = login(client)
    no = client.get(f"/admin/predictions/{pid_no}", headers=auth(token)).json()["explain"]
    yes = client.get(f"/admin/predictions/{pid_yes}", headers=auth(token)).json()["explain"]
    assert not any("heatmap_base64" in v for v in no.values())
    assert yes["cnn-transfer"]["heatmap_base64"] == HEATMAP


def test_no_image_written_without_consent(client, store, mock_models):
    for consent in (None, False):
        r = do_predict(client, consent=consent)
        assert r.status_code == 200
        assert r.json()["prediction_id"]
    assert all_files(store.uploads_dir) == []
    rows = store._conn().execute("SELECT stored_image_path, consent_store_image FROM predictions").fetchall()
    assert len(rows) == 2
    assert all(r["stored_image_path"] is None and r["consent_store_image"] == 0 for r in rows)
    # No image bytes anywhere in the data dir (DB holds metadata only)
    for f in all_files(store.data_dir):
        assert b"SecretCameraMaker" not in f.read_bytes()

    token = login(client)
    item = client.get("/admin/predictions", headers=auth(token)).json()["items"][0]
    r = client.get(f"/admin/predictions/{item['id']}/image", headers=auth(token))
    assert r.status_code == 404


def test_consented_image_is_webp_without_exif(client, store, mock_models):
    r = do_predict(client, consent=True, image_bytes=jpeg_with_exif((80, 60)))
    pid = r.json()["prediction_id"]

    files = all_files(store.uploads_dir)
    assert len(files) == 1
    path = files[0]
    assert path.name == f"{pid}.webp"
    assert path.parent.name == utcnow().strftime("%Y-%m")
    raw = path.read_bytes()
    assert b"SecretCameraMaker" not in raw and b"SecretModel" not in raw and b"Exif" not in raw
    with Image.open(path) as img:
        assert img.format == "WEBP"
        assert img.size == (80, 60)
        assert "exif" not in img.info and "icc_profile" not in img.info
        assert len(img.getexif()) == 0

    token = login(client)
    r = client.get(f"/admin/predictions/{pid}/image", headers=auth(token))
    assert r.status_code == 200 and r.headers["content-type"] == "image/webp"
    r = client.get(f"/admin/predictions/{pid}/image?thumb=32", headers=auth(token))
    with Image.open(io.BytesIO(r.content)) as thumb:
        assert max(thumb.size) <= 32

    item = client.get("/admin/predictions?has_image=true", headers=auth(token)).json()["items"][0]
    assert item["has_image"] is True

    # Delete removes both the row and the file
    r = client.delete(f"/admin/predictions/{pid}", headers=auth(token))
    assert r.status_code == 204
    assert not path.exists()
    assert client.get(f"/admin/predictions/{pid}", headers=auth(token)).status_code == 404


def test_sanitize_applies_orientation_and_downscales():
    img = Image.new("RGB", (40, 20), (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 CW on display
    exif[0x010F] = "SecretCameraMaker"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif.tobytes())
    out = sanitize_image_for_storage(buf.getvalue(), max_side=10, quality=80)
    assert b"SecretCameraMaker" not in out
    with Image.open(io.BytesIO(out)) as res:
        assert res.format == "WEBP"
        assert res.size == (5, 10)  # rotated to portrait, then downscaled


def test_store_failure_does_not_break_predict(client, store, mock_models, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("disk full")
    monkeypatch.setattr(store, "record_prediction", boom)
    r = do_predict(client, consent=True)
    assert r.status_code == 200
    assert r.json()["prediction_id"] is None
    assert all_files(store.uploads_dir) == []

    monkeypatch.setattr(routes_predict, "get_store", MagicMock(side_effect=RuntimeError("no db")))
    r = do_predict(client)
    assert r.status_code == 200 and r.json()["prediction_id"] is None


def test_store_disabled(client, mock_models, monkeypatch):
    monkeypatch.setattr(settings, "STORE_ENABLED", False)
    r = do_predict(client)
    assert r.status_code == 200 and r.json()["prediction_id"] is None
    r = client.post("/feedback", json={"prediction_id": "0" * 32, "user_knows_truth": True, "user_label": "ai"})
    assert r.status_code == 503


# ------------------------------------------------------------------- auth

ADMIN_ENDPOINTS = [
    ("GET", "/admin/session"),
    ("GET", "/admin/predictions"),
    ("GET", "/admin/predictions/abc"),
    ("GET", "/admin/predictions/abc/image"),
    ("DELETE", "/admin/predictions/abc"),
    ("GET", "/admin/stats"),
    ("GET", "/admin/export.csv"),
    ("POST", "/admin/purge"),
]


@pytest.mark.parametrize("method,path", ADMIN_ENDPOINTS)
def test_admin_requires_token(client, store, method, path):
    assert client.request(method, path).status_code == 401
    for header in ("Bearer", "Bearer ", "Basic YWRtaW46eA==", "Bearer not.a.token", "bearer x"):
        r = client.request(method, path, headers={"Authorization": header})
        assert r.status_code == 401, header
    # Cookies are never accepted as credentials
    token, _ = issue_token()
    client.cookies.set("access_token", token)
    assert client.request(method, path).status_code == 401
    client.cookies.clear()


def test_expired_and_forged_tokens_rejected(client, store, monkeypatch):
    good, _ = issue_token()
    assert client.get("/admin/session", headers=auth(good)).status_code == 200

    expired, _ = issue_token(now=time.time() - 7200, ttl=60)
    assert verify_token(expired) is None
    assert client.get("/admin/session", headers=auth(expired)).status_code == 401

    # Tamper with the payload (extend expiry) but keep the old signature
    payload_b64, sig = good.split(".")
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    payload["exp"] += 10 ** 6
    forged_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    forged = f"{forged_payload}.{sig}"
    assert client.get("/admin/session", headers=auth(forged)).status_code == 401

    # Signed with a different secret
    monkeypatch.setattr(settings, "ADMIN_TOKEN_SECRET", "attacker-secret")
    other, _ = issue_token()
    monkeypatch.setattr(settings, "ADMIN_TOKEN_SECRET", "test-token-secret")
    assert client.get("/admin/session", headers=auth(other)).status_code == 401

    # Rotating the password invalidates outstanding tokens
    monkeypatch.setattr(settings, "ADMIN_PASSWORD_HASH", hash_password("another long password", n=2 ** 12))
    assert client.get("/admin/session", headers=auth(good)).status_code == 401


def test_admin_disabled_without_password_hash(client, store, monkeypatch):
    token, _ = issue_token()
    monkeypatch.setattr(settings, "ADMIN_PASSWORD_HASH", None)
    assert client.post("/admin/login", json={"password": PASSWORD}).status_code == 503
    assert client.get("/admin/predictions", headers=auth(token)).status_code == 401


def test_login_wrong_password_and_lockout(client, store, monkeypatch):
    monkeypatch.setattr(settings, "ADMIN_LOGIN_MAX_FAILURES", 3)
    for _ in range(3):
        r = client.post("/admin/login", json={"password": "wrong"})
        assert r.status_code == 401
    r = client.post("/admin/login", json={"password": PASSWORD})
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) > 0


def test_password_hash_roundtrip():
    encoded = hash_password("s3cret-password!", n=2 ** 12)
    assert encoded.startswith("scrypt$4096$8$1$")
    assert verify_password("s3cret-password!", encoded)
    assert not verify_password("s3cret-password?", encoded)
    assert not verify_password("anything", "not-a-hash")
    assert not verify_password("anything", None)


# ------------------------------------------------------------- pagination

def test_cursor_pagination_and_filters(client, store):
    now = utcnow()
    ids = []
    for i, prob in enumerate([0.1, 0.5, 0.9, 0.55, 0.95]):
        ids.append(store.record_prediction(
            image_bytes=f"img-{i}".encode(), final_pred="fake" if prob >= 0.5 else "real",
            final_prob=prob, fusion_used=True, created_at=now - timedelta(minutes=10 - i),
            variant="prod" if i % 2 == 0 else "candidate",
        ))
    token = login(client)
    seen, cursor = [], None
    while True:
        url = "/admin/predictions?limit=2" + (f"&cursor={cursor}" if cursor else "")
        page = client.get(url, headers=auth(token)).json()
        seen += [i["id"] for i in page["items"]]
        cursor = page["next_cursor"]
        if not cursor:
            break
    assert seen == list(reversed(ids))

    r = client.get("/admin/predictions?uncertain=true", headers=auth(token)).json()
    assert {i["id"] for i in r["items"]} == {ids[1], ids[3]}
    r = client.get("/admin/predictions?variant=candidate", headers=auth(token)).json()
    assert {i["id"] for i in r["items"]} == {ids[1], ids[3]}
    today = now.strftime("%Y-%m-%d")
    r = client.get(f"/admin/predictions?date_from={today}&date_to={today}", headers=auth(token)).json()
    assert len(r["items"]) == 5
    r = client.get("/admin/predictions?date_to=2000-01-01", headers=auth(token)).json()
    assert r["items"] == []
    assert client.get("/admin/predictions?cursor=%%%", headers=auth(token)).status_code == 400

    stats = client.get("/admin/stats", headers=auth(token)).json()
    variants = {v["variant"]: v for v in stats["per_variant"]}
    assert variants["prod"]["predictions"] == 3 and variants["candidate"]["predictions"] == 2


def test_csv_export_neutralises_formulas(client, store):
    pid = store.record_prediction(image_bytes=b"x", final_pred="real", final_prob=0.2, fusion_used=True)
    store.upsert_feedback(prediction_id=pid, user_knows_truth=True, user_label="real",
                          source_note="=HYPERLINK(\"http://evil\")")
    token = login(client)
    text = client.get("/admin/export.csv", headers=auth(token)).text
    assert "'=HYPERLINK" in text


# ---------------------------------------------------------------- retention

def test_purge_removes_old_rows_and_files(client, store, mock_models):
    old_id = store.record_prediction(
        image_bytes=jpeg_with_exif(), final_pred="fake", final_prob=0.9, fusion_used=True,
        consent_store_image=True, created_at=utcnow() - timedelta(days=120),
    )
    assert store.save_consented_image(old_id, jpeg_with_exif())
    store.upsert_feedback(prediction_id=old_id, user_knows_truth=True, user_label="ai")
    new_id = do_predict(client, consent=True).json()["prediction_id"]
    assert len(all_files(store.uploads_dir)) == 2

    removed = store.purge(retention_days=90)
    assert removed["predictions"] == 1 and removed["images"] == 1
    assert store.get_prediction(old_id) is None
    assert store.get_prediction(new_id) is not None
    assert store._conn().execute("SELECT COUNT(*) FROM feedback").fetchone()[0] == 0
    remaining = all_files(store.uploads_dir)
    assert [p.stem for p in remaining] == [new_id]

    # Admin-triggered purge uses RETENTION_DAYS
    token = login(client)
    r = client.post("/admin/purge", headers=auth(token))
    assert r.status_code == 200 and r.json()["retention_days"] == settings.RETENTION_DAYS


# --------------------------------------------------------------- rate limit

def test_rate_limit_returns_429(client, store, mock_models, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_PREDICT", "2/minute")
    monkeypatch.setattr(settings, "RATE_LIMIT_FEEDBACK", "1/minute")
    monkeypatch.setattr(settings, "RATE_LIMIT_ADMIN_LOGIN", "1/minute")
    assert do_predict(client).status_code == 200
    pid = do_predict(client).json()["prediction_id"]
    r = do_predict(client)
    assert r.status_code == 429
    assert int(r.headers["retry-after"]) >= 1

    fb = {"prediction_id": pid, "user_knows_truth": True, "user_label": "ai"}
    assert client.post("/feedback", json=fb).status_code == 200
    assert client.post("/feedback", json=fb).status_code == 429

    assert client.post("/admin/login", json={"password": "wrong"}).status_code == 401
    assert client.post("/admin/login", json={"password": PASSWORD}).status_code == 429


def test_rate_limit_explain_model(client, monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", True)
    monkeypatch.setattr(settings, "RATE_LIMIT_EXPLAIN", "1/hour")
    form = {"model_name": "cnn-transfer", "prob_fake": "0.5"}
    files = {"image": ("a.jpg", io.BytesIO(jpeg_with_exif()), "image/jpeg")}
    first = client.post("/explain-model", data=form, files=files)
    assert first.status_code != 429
    files = {"image": ("a.jpg", io.BytesIO(jpeg_with_exif()), "image/jpeg")}
    assert client.post("/explain-model", data=form, files=files).status_code == 429


def test_token_bucket_refills():
    limiter = rate_limit.TokenBucketLimiter()
    assert limiter.hit("k", 2, 60, now=0)[0]
    assert limiter.hit("k", 2, 60, now=0)[0]
    allowed, retry = limiter.hit("k", 2, 60, now=0)
    assert not allowed and retry == pytest.approx(30)
    assert limiter.hit("k", 2, 60, now=31)[0]
    assert rate_limit.parse_rate("10/minute") == (10.0, 60.0)
    assert rate_limit.parse_rate("5/30") == (5.0, 30.0)
    assert rate_limit.parse_rate("off") is None


def _request(peer: str, xff: str = None) -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff else []
    return Request({"type": "http", "client": (peer, 1234), "headers": headers})


def test_forwarded_for_only_from_trusted_proxy(monkeypatch):
    monkeypatch.setattr(settings, "TRUSTED_PROXIES", "")
    assert rate_limit.get_client_ip(_request("10.0.0.5", "1.2.3.4")) == "10.0.0.5"

    monkeypatch.setattr(settings, "TRUSTED_PROXIES", "10.0.0.0/8")
    # Peer is trusted: walk right to left, skipping trusted hops
    assert rate_limit.get_client_ip(_request("10.0.0.5", "6.6.6.6, 1.2.3.4, 10.1.1.1")) == "1.2.3.4"
    # Peer is not a trusted proxy: header ignored
    assert rate_limit.get_client_ip(_request("8.8.8.8", "1.2.3.4")) == "8.8.8.8"

    monkeypatch.setattr(settings, "TRUSTED_PROXIES", "*")
    # Spoofed left-most entry is ignored; the hop appended by the proxy wins
    assert rate_limit.get_client_ip(_request("172.16.0.1", "6.6.6.6, 1.2.3.4")) == "1.2.3.4"
    assert rate_limit.get_client_ip(_request("172.16.0.1")) == "172.16.0.1"


def test_client_ip_is_hashed_not_stored(client, store, mock_models):
    do_predict(client)
    row = store._conn().execute("SELECT client_hash FROM predictions").fetchone()
    assert row["client_hash"] and "testclient" not in row["client_hash"]
    assert len(row["client_hash"]) == 32
