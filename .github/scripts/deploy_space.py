#!/usr/bin/env python3
"""Deploy backend/ to the Hugging Face Space (CI counterpart of backend/deploy-to-hf.sh).

Steps:
  1. Stage backend/ into a temp directory, skipping the exact ``exclude_items``
     list parsed from backend/deploy-to-hf.sh (single source of truth).
  2. Rewrite ``ARG BASE_IMAGE`` / ``ARG INSTALL_OPTIONAL_DEPS`` in the staged
     Dockerfile, like the manual script does.
  3. Compare the staged model-manifest.json with the copy currently in the Space.
  4. Upload the staged folder as one commit with ``HfApi.upload_folder``;
     ``delete_patterns="*"`` removes Space files that no longer exist in backend/.
  5. Restart the Space only when the manifest changed and the upload did not
     create a new commit (a new commit already triggers a rebuild), or when
     FORCE_RESTART=true and the manifest changed.

Environment:
  HF_TOKEN               write token for the Space (required unless --dry-run)
  HF_SPACE_ID            "<owner>/<space-name>" (required unless --dry-run)
  HF_BASE_IMAGE          base image for the app Dockerfile (default: python:3.11-slim)
  INSTALL_OPTIONAL_DEPS  "true"/"false" (default: false)
  FORCE_RESTART          "true" to restart after a manifest change even if a new commit was made
  GITHUB_SHA, GITHUB_OUTPUT  provided by GitHub Actions
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_ROOT / "backend"
DEPLOY_SCRIPT = BACKEND_DIR / "deploy-to-hf.sh"
MANIFEST_NAME = "model-manifest.json"

# Cache directories that should never be shipped, at any depth.
NESTED_EXCLUDES = {"__pycache__", ".pytest_cache"}


def parse_excludes(script: Path = DEPLOY_SCRIPT) -> list[str]:
    """Read the ``exclude_items=( ... )`` bash array from deploy-to-hf.sh."""
    text = script.read_text(encoding="utf-8")
    match = re.search(r"exclude_items=\(\s*(.*?)\)", text, re.DOTALL)
    if not match:
        raise SystemExit(f"Could not find exclude_items in {script}")
    items = re.findall(r'"([^"]+)"', match.group(1))
    if not items:
        raise SystemExit(f"exclude_items in {script} is empty or unparsable")
    return items


def stage_backend(dest: Path, excludes: list[str]) -> list[str]:
    """Copy top-level backend items not in ``excludes`` into ``dest``."""
    copied: list[str] = []
    ignore = shutil.ignore_patterns(*NESTED_EXCLUDES)
    for path in sorted(BACKEND_DIR.iterdir()):
        if path.name in excludes:
            continue
        target = dest / path.name
        if path.is_dir():
            shutil.copytree(path, target, ignore=ignore)
        else:
            shutil.copy2(path, target)
        copied.append(path.name)
    return copied


def patch_dockerfile(dockerfile: Path, base_image: str, optional_deps: str) -> None:
    if not dockerfile.exists():
        return
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        if line.startswith("ARG BASE_IMAGE="):
            line = f"ARG BASE_IMAGE={base_image}"
        elif line.startswith("ARG INSTALL_OPTIONAL_DEPS="):
            line = f"ARG INSTALL_OPTIONAL_DEPS={optional_deps}"
        out.append(line)
    dockerfile.write_text("\n".join(out) + "\n", encoding="utf-8")


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def remote_manifest(api, space_id: str):
    """Return the manifest currently in the Space, or None if absent."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

    try:
        path = hf_hub_download(
            repo_id=space_id,
            repo_type="space",
            filename=MANIFEST_NAME,
            token=api.token,
        )
    except (EntryNotFoundError, RepositoryNotFoundError):
        return None
    return load_json(Path(path))


def write_outputs(**values: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    for key, value in values.items():
        print(f"{key}={value}")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            for key, value in values.items():
                fh.write(f"{key}={value}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="stage and list files only")
    args = parser.parse_args()

    base_image = os.environ.get("HF_BASE_IMAGE") or "python:3.11-slim"
    optional_deps = (os.environ.get("INSTALL_OPTIONAL_DEPS") or "false").lower()
    if optional_deps not in {"true", "false"}:
        print("WARNING: INSTALL_OPTIONAL_DEPS must be true or false; using false")
        optional_deps = "false"
    force_restart = (os.environ.get("FORCE_RESTART") or "false").lower() == "true"

    excludes = parse_excludes()
    print(f"Excludes (from {DEPLOY_SCRIPT.name}): {', '.join(excludes)}")

    with tempfile.TemporaryDirectory(prefix="hf-space-") as tmp:
        stage = Path(tmp)
        copied = stage_backend(stage, excludes)
        patch_dockerfile(stage / "Dockerfile", base_image, optional_deps)
        print(f"Staged: {', '.join(copied)}")
        print(f"Docker BASE_IMAGE={base_image} INSTALL_OPTIONAL_DEPS={optional_deps}")

        if args.dry_run:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    print(f"  {p.relative_to(stage).as_posix()}")
            return 0

        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        space_id = os.environ.get("HF_SPACE_ID")
        if not token or not space_id:
            print("ERROR: HF_TOKEN and HF_SPACE_ID must be set")
            return 1

        api = HfApi(token=token)
        local_manifest = load_json(stage / MANIFEST_NAME)
        manifest_changed = local_manifest is not None and local_manifest != remote_manifest(api, space_id)
        sha_before = api.space_info(space_id).sha

        sha = os.environ.get("GITHUB_SHA", "local")[:7]
        commit = api.upload_folder(
            repo_id=space_id,
            repo_type="space",
            folder_path=str(stage),
            path_in_repo=".",
            delete_patterns="*",
            commit_message=f"Deploy backend from GitHub {sha}",
            commit_description=f"Automated deploy of backend/ at commit {os.environ.get('GITHUB_SHA', 'unknown')}.",
        )
        sha_after = api.space_info(space_id).sha
        new_commit = sha_after != sha_before
        print(f"Upload result: {getattr(commit, 'commit_url', commit)}")
        print(f"Space revision: {sha_before} -> {sha_after} (new commit: {new_commit})")
        print(f"Manifest changed: {manifest_changed}")

    restarted = False
    if manifest_changed and (not new_commit or force_restart):
        print("Restarting Space so it boots with the new model manifest...")
        api.restart_space(space_id)
        restarted = True
    elif manifest_changed:
        print("Manifest changed; the new commit triggers a rebuild, so no explicit restart is needed.")

    write_outputs(
        manifest_changed=str(manifest_changed).lower(),
        new_commit=str(new_commit).lower(),
        restarted=str(restarted).lower(),
        space_sha=sha_after or "",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
