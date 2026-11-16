#!/usr/bin/env python3
"""Pin a model repo to a revision in backend/model-manifest.json.

Usage:
  python .github/scripts/bump_manifest.py --repo-id DeepFakeDetector/vit-base-final [--revision <sha|branch|tag>]

A missing revision, or one that is not a full 40-char commit sha, is resolved
to the commit sha it currently points at on the Hub, so the manifest always
pins an immutable revision. The repo must already be listed in the manifest
(either as ``fusion`` or under ``submodels``); unknown repos are rejected.

Writes ``changed``, ``repo_id``, ``old_revision`` and ``revision`` to
$GITHUB_OUTPUT when running in GitHub Actions.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[2] / "backend" / "model-manifest.json"
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def resolve_revision(repo_id: str, revision: str | None) -> str:
    if revision and FULL_SHA.match(revision):
        return revision
    from huggingface_hub import HfApi

    info = HfApi(token=os.environ.get("HF_TOKEN") or None).model_info(repo_id, revision=revision or None)
    if not info.sha:
        raise SystemExit(f"Could not resolve a commit sha for {repo_id}@{revision or 'main'}")
    return info.sha


def find_entry(manifest: dict, repo_id: str) -> dict | None:
    entries = [manifest.get("fusion") or {}] + list(manifest.get("submodels") or [])
    for entry in entries:
        if entry.get("repo_id") == repo_id:
            return entry
    return None


def write_outputs(**values: str) -> None:
    out = os.environ.get("GITHUB_OUTPUT")
    for key, value in values.items():
        print(f"{key}={value}")
    if out:
        with open(out, "a", encoding="utf-8") as fh:
            for key, value in values.items():
                fh.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bump a pinned revision in model-manifest.json")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    args = parser.parse_args(argv)

    repo_id = args.repo_id.strip()
    if not REPO_ID.match(repo_id):
        print(f"ERROR: invalid repo id: {repo_id!r}")
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entry = find_entry(manifest, repo_id)
    if entry is None:
        print(f"ERROR: {repo_id} is not listed in {args.manifest.name}; refusing to add it automatically")
        return 1

    revision = resolve_revision(repo_id, args.revision.strip() or None)
    old = entry.get("revision")
    changed = old != revision
    if changed:
        entry["revision"] = revision
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Pinned {repo_id}: {old or 'main (unpinned)'} -> {revision}")
    else:
        print(f"{repo_id} is already pinned to {revision}")

    write_outputs(
        changed=str(changed).lower(),
        repo_id=repo_id,
        old_revision=old or "",
        revision=revision,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
