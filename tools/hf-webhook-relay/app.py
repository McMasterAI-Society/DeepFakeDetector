"""Hugging Face webhook -> GitHub repository_dispatch relay.

Receives HF Hub webhooks for the DeepFakeDetector org, keeps only content
updates to allowlisted model repos, and forwards them to GitHub as a
``repository_dispatch`` event (default type ``hf-model-updated``) that the
``model-update.yml`` workflow listens for.

Configuration (environment variables):
  WEBHOOK_SECRET      secret set on the HF webhook (sent as X-Webhook-Secret)   [required]
  GITHUB_TOKEN        fine-grained PAT with "Contents: read and write" on GITHUB_REPO [required]
  GITHUB_REPO         target repo, e.g. McMasterAI-Society/DeepFakeDetector      [required]
  ALLOWED_REPOS       comma-separated model repo ids to forward (default: the production manifest repos)
  DISPATCH_EVENT_TYPE repository_dispatch event type (default: hf-model-updated)
  GITHUB_API_URL      default https://api.github.com
"""

from __future__ import annotations

import hmac
import logging
import os
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI, Header, HTTPException, Request

logger = logging.getLogger("hf-webhook-relay")

DEFAULT_ALLOWED_REPOS = (
    "DeepFakeDetector/fusion-logreg-final",
    "DeepFakeDetector/cnn-transfer-final",
    "DeepFakeDetector/vit-base-final",
    "DeepFakeDetector/deit-distilled-final",
    "DeepFakeDetector/gradfield-cnn-final",
)


@dataclass(frozen=True)
class Settings:
    webhook_secret: str
    github_token: str
    github_repo: str
    allowed_repos: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_ALLOWED_REPOS))
    event_type: str = "hf-model-updated"
    github_api_url: str = "https://api.github.com"

    @classmethod
    def from_env(cls) -> "Settings":
        allowed = os.environ.get("ALLOWED_REPOS", "")
        repos = frozenset(r.strip() for r in allowed.split(",") if r.strip()) or frozenset(DEFAULT_ALLOWED_REPOS)
        return cls(
            webhook_secret=os.environ.get("WEBHOOK_SECRET", ""),
            github_token=os.environ.get("GITHUB_TOKEN", ""),
            github_repo=os.environ.get("GITHUB_REPO", ""),
            allowed_repos=repos,
            event_type=os.environ.get("DISPATCH_EVENT_TYPE", "hf-model-updated"),
            github_api_url=os.environ.get("GITHUB_API_URL", "https://api.github.com").rstrip("/"),
        )


def secret_matches(expected: str, provided: str | None) -> bool:
    """Constant-time comparison; an unset expected secret never matches."""
    if not expected or provided is None:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), provided.encode("utf-8"))


def extract_update(payload: dict, allowed_repos: frozenset[str]) -> tuple[dict | None, str]:
    """Return (client_payload, reason). client_payload is None when the event is ignored."""
    event = payload.get("event") or {}
    repo = payload.get("repo") or {}

    if repo.get("type") != "model":
        return None, "not a model repo"
    scope = str(event.get("scope") or "")
    if not scope.startswith("repo.content"):
        return None, f"ignored scope {scope!r}"
    if event.get("action") == "delete":
        return None, "ignored delete action"

    repo_id = repo.get("name")
    if repo_id not in allowed_repos:
        return None, f"repo {repo_id!r} is not allowlisted"

    # Only forward pushes to main when HF tells us which refs moved.
    refs = payload.get("updatedRefs")
    if isinstance(refs, list) and refs:
        if not any(isinstance(r, dict) and r.get("ref") == "refs/heads/main" for r in refs):
            return None, "main was not updated"

    revision = repo.get("headSha")
    return {"repo_id": repo_id, "revision": revision}, "forwarded"


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="HF webhook relay", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/")
    async def health() -> dict:
        configured = bool(settings.webhook_secret and settings.github_token and settings.github_repo)
        return {"status": "ok", "configured": configured}

    @app.post("/webhook")
    async def webhook(request: Request, x_webhook_secret: str | None = Header(default=None)) -> dict:
        if not secret_matches(settings.webhook_secret, x_webhook_secret):
            raise HTTPException(status_code=401, detail="invalid webhook secret")

        try:
            payload = await request.json()
        except ValueError:
            raise HTTPException(status_code=400, detail="invalid JSON body")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid payload")

        client_payload, reason = extract_update(payload, settings.allowed_repos)
        if client_payload is None:
            logger.info("Ignored webhook: %s", reason)
            return {"forwarded": False, "reason": reason}

        if not settings.github_token or not settings.github_repo:
            raise HTTPException(status_code=500, detail="relay is not configured")

        url = f"{settings.github_api_url}/repos/{settings.github_repo}/dispatches"
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {settings.github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "hf-webhook-relay",
        }
        body = {"event_type": settings.event_type, "client_payload": client_payload}
        async with httpx.AsyncClient(transport=transport, timeout=15.0) as client:
            try:
                resp = await client.post(url, json=body, headers=headers)
            except httpx.HTTPError as exc:
                logger.error("GitHub dispatch failed: %s", exc)
                raise HTTPException(status_code=502, detail="GitHub dispatch failed")
        if resp.status_code != 204:
            logger.error("GitHub dispatch returned %s: %s", resp.status_code, resp.text[:500])
            raise HTTPException(status_code=502, detail=f"GitHub dispatch returned {resp.status_code}")

        logger.info("Forwarded %s@%s", client_payload["repo_id"], client_payload["revision"])
        return {"forwarded": True, **client_payload}

    return app


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

# `uvicorn app:app` entry point. Settings are read from the environment at import;
# with no WEBHOOK_SECRET set, every webhook is rejected with 401.
app = create_app()
