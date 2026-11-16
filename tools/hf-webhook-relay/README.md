---
title: HF Webhook Relay
emoji: 🔁
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# HF webhook relay

A small FastAPI app that turns Hugging Face Hub webhooks into GitHub
`repository_dispatch` events. It is what lets a new commit on a model repo in the
`DeepFakeDetector` org open a "pin new revision" PR in GitHub
(`.github/workflows/model-update.yml`).

```
HF model repo push ──webhook──> this relay ──repository_dispatch(hf-model-updated)──> GitHub
                                                                              └─> model-update.yml opens a PR
```

The relay forwards an event only when all of these hold:

1. `X-Webhook-Secret` equals `WEBHOOK_SECRET` (compared in constant time; an unset secret rejects everything with 401).
2. `repo.type == "model"`.
3. `event.scope` starts with `repo.content` and `event.action` is not `delete`.
4. `repo.name` is in `ALLOWED_REPOS`.
5. If the payload lists `updatedRefs`, one of them is `refs/heads/main`.

The dispatched `client_payload` is `{"repo_id": "<repo.name>", "revision": "<repo.headSha>"}`.
Ignored events get a `200` with `{"forwarded": false, "reason": ...}` so HF does not retry them.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `WEBHOOK_SECRET` | yes | Same value as the secret on the HF webhook. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"`. |
| `GITHUB_TOKEN` | yes | Fine-grained PAT, resource owner `McMasterAI-Society`, repository access limited to `DeepFakeDetector`, permission **Contents: Read and write** (required by the dispatches API). Nothing else. |
| `GITHUB_REPO` | yes | `McMasterAI-Society/DeepFakeDetector` |
| `ALLOWED_REPOS` | no | Comma-separated model repo ids. Defaults to the five repos in `backend/model-manifest.json`. Keep the two in sync. |
| `DISPATCH_EVENT_TYPE` | no | Defaults to `hf-model-updated`. |
| `LOG_LEVEL` | no | Defaults to `INFO`. |

## Setup

1. **Create the Space.** On huggingface.co, create a new Space under the `DeepFakeDetector` org
   (for example `DeepFakeDetector/hf-webhook-relay`), SDK **Docker**, hardware **CPU basic**, visibility
   **public**: HF webhooks call the URL without HF credentials, so a private Space would reject them.
   The endpoint is protected by the webhook secret.
2. **Upload this folder** to the Space (only these files are needed: `README.md`, `Dockerfile`,
   `requirements.txt`, `app.py`):
   ```bash
   hf upload DeepFakeDetector/hf-webhook-relay tools/hf-webhook-relay . --repo-type space \
     --exclude "tests/*" --exclude "requirements-dev.txt"
   ```
3. **Set Space secrets** (Settings -> Variables and secrets): `WEBHOOK_SECRET`, `GITHUB_TOKEN`,
   and the variable `GITHUB_REPO`. Restart the Space. `GET /` should return `"configured": true`.
4. **Create the webhook** at <https://huggingface.co/settings/webhooks>:
   - Target repositories: the `DeepFakeDetector` org (or the individual model repos).
   - Webhook URL: `https://<space-subdomain>.hf.space/webhook`
   - Secret: the `WEBHOOK_SECRET` value.
   - Triggers: **Repo update** (content). Discussions are not needed.
5. **Test it.** Push a trivial commit (for example a README edit) to an allowlisted model repo. The
   webhook's activity log on HF should show a `200` with `"forwarded": true`, and a
   **Model update PR** run should start in GitHub Actions.

To pin a revision without the relay, run the **Model update PR** workflow by hand
(Actions -> Model update PR -> Run workflow) with `repo_id` and optionally `revision`.

## Running locally

```bash
cd tools/hf-webhook-relay
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
WEBHOOK_SECRET=dev GITHUB_TOKEN=... GITHUB_REPO=owner/repo uvicorn app:app --port 7860
```

## Security notes

- The PAT can only write repo contents on one repo; rotate it if the Space is ever compromised.
- The relay never trusts the payload beyond the allowlist. `model-update.yml` re-checks that the repo is in
  `backend/model-manifest.json` and resolves the revision to a commit sha on the Hub before opening a PR.
- The PR never deploys by itself. Merging it triggers `deploy-space.yml`, which waits for approval on the
  `production` environment.
