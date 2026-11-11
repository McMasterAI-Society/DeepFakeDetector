import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import Settings, create_app, secret_matches  # noqa: E402

SECRET = "s3cret-value"
REPO = "DeepFakeDetector/vit-base-final"
SHA = "a" * 40


def make_payload(**overrides):
    payload = {
        "event": {"action": "update", "scope": "repo.content"},
        "repo": {"type": "model", "name": REPO, "headSha": SHA},
        "updatedRefs": [{"ref": "refs/heads/main", "oldSha": "b" * 40, "newSha": SHA}],
        "webhook": {"id": "x", "version": 3},
    }
    for key, value in overrides.items():
        payload[key] = value
    return payload


class Recorder:
    def __init__(self, status=204):
        self.status = status
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status)


@pytest.fixture
def recorder():
    return Recorder()


@pytest.fixture
def client(recorder):
    settings = Settings(webhook_secret=SECRET, github_token="ghp_test", github_repo="org/repo")
    app = create_app(settings, transport=httpx.MockTransport(recorder))
    return TestClient(app)


def post(client, payload, secret=SECRET):
    headers = {} if secret is None else {"X-Webhook-Secret": secret}
    return client.post("/webhook", json=payload, headers=headers)


def test_forwards_allowlisted_model_update(client, recorder):
    resp = post(client, make_payload())
    assert resp.status_code == 200
    assert resp.json() == {"forwarded": True, "repo_id": REPO, "revision": SHA}

    assert len(recorder.requests) == 1
    req = recorder.requests[0]
    assert req.method == "POST"
    assert str(req.url) == "https://api.github.com/repos/org/repo/dispatches"
    assert req.headers["Authorization"] == "Bearer ghp_test"
    assert json.loads(req.content) == {
        "event_type": "hf-model-updated",
        "client_payload": {"repo_id": REPO, "revision": SHA},
    }


@pytest.mark.parametrize("secret", [None, "", "wrong", SECRET + "x"])
def test_rejects_bad_secret(client, recorder, secret):
    resp = post(client, make_payload(), secret=secret)
    assert resp.status_code == 401
    assert recorder.requests == []


def test_unset_secret_rejects_everything(recorder):
    settings = Settings(webhook_secret="", github_token="t", github_repo="org/repo")
    client = TestClient(create_app(settings, transport=httpx.MockTransport(recorder)))
    assert post(client, make_payload(), secret="").status_code == 401
    assert recorder.requests == []


@pytest.mark.parametrize(
    "payload, reason",
    [
        (make_payload(repo={"type": "space", "name": REPO, "headSha": SHA}), "not a model repo"),
        (make_payload(repo={"type": "dataset", "name": REPO, "headSha": SHA}), "not a model repo"),
        (make_payload(event={"action": "update", "scope": "repo.config"}), "ignored scope"),
        (make_payload(event={"action": "create", "scope": "discussion"}), "ignored scope"),
        (make_payload(event={"action": "delete", "scope": "repo.content"}), "ignored delete"),
        (
            make_payload(repo={"type": "model", "name": "someone/else", "headSha": SHA}),
            "not allowlisted",
        ),
        (make_payload(updatedRefs=[{"ref": "refs/heads/dev"}]), "main was not updated"),
    ],
)
def test_ignores_irrelevant_events(client, recorder, payload, reason):
    resp = post(client, payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["forwarded"] is False
    assert reason in body["reason"]
    assert recorder.requests == []


def test_accepts_scope_subtypes_and_missing_refs(client, recorder):
    payload = make_payload(event={"action": "update", "scope": "repo.content.something"})
    payload.pop("updatedRefs")
    assert post(client, payload).json()["forwarded"] is True
    assert len(recorder.requests) == 1


def test_github_error_returns_502():
    rec = Recorder(status=404)
    settings = Settings(webhook_secret=SECRET, github_token="t", github_repo="org/repo")
    client = TestClient(create_app(settings, transport=httpx.MockTransport(rec)))
    assert post(client, make_payload()).status_code == 502


def test_network_error_returns_502():
    def boom(request):
        raise httpx.ConnectError("down", request=request)

    settings = Settings(webhook_secret=SECRET, github_token="t", github_repo="org/repo")
    client = TestClient(create_app(settings, transport=httpx.MockTransport(boom)))
    assert post(client, make_payload()).status_code == 502


def test_invalid_json_returns_400(client):
    resp = client.post(
        "/webhook", content=b"not json", headers={"X-Webhook-Secret": SECRET, "Content-Type": "application/json"}
    )
    assert resp.status_code == 400


def test_custom_allowlist_and_event_type(recorder):
    settings = Settings(
        webhook_secret=SECRET,
        github_token="t",
        github_repo="org/repo",
        allowed_repos=frozenset({"someone/else"}),
        event_type="custom",
    )
    client = TestClient(create_app(settings, transport=httpx.MockTransport(recorder)))
    payload = make_payload(repo={"type": "model", "name": "someone/else", "headSha": SHA})
    assert post(client, payload).json()["forwarded"] is True
    assert json.loads(recorder.requests[0].content)["event_type"] == "custom"
    assert post(client, make_payload()).json()["forwarded"] is False


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("WEBHOOK_SECRET", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPO", "org/repo")
    monkeypatch.setenv("ALLOWED_REPOS", " a/b , c/d ,")
    s = Settings.from_env()
    assert s.allowed_repos == frozenset({"a/b", "c/d"})
    monkeypatch.delenv("ALLOWED_REPOS")
    assert REPO in Settings.from_env().allowed_repos


def test_secret_matches():
    assert secret_matches("abc", "abc")
    assert not secret_matches("abc", "abd")
    assert not secret_matches("", "")
    assert not secret_matches("abc", None)


def test_health(client):
    assert client.get("/").json() == {"status": "ok", "configured": True}
