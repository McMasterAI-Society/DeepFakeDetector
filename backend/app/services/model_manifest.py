"""
Model manifest handling: which repos, at which revisions, with which artifacts.

The fusion repo's ``config.json`` lists its submodels. Each entry may be a plain
repo id string (legacy) or an object::

    {"repo_id": "...", "revision": "<sha>", "weights_file": "...", "input_size": 224}

An optional local ``model-manifest.json`` can pin revisions on top of that::

    {"fusion": {"repo_id": "...", "revision": "<sha>"},
     "submodels": [{"repo_id": "...", "revision": "<sha>"}]}
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.core.errors import ConfigurationError
from app.core.logging import get_logger

logger = get_logger(__name__)

# backend/ (parent of the app package)
BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = BACKEND_DIR / "model-manifest.json"


@dataclass
class SubmodelSpec:
    """Resolved description of one submodel to load."""

    repo_id: str
    revision: Optional[str] = None
    weights_file: Optional[str] = None
    input_size: Optional[int] = None


def _normalize_input_size(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    return int(value)


def parse_submodel_entries(entries: Iterable[Any]) -> List[SubmodelSpec]:
    """
    Parse the fusion config ``submodels`` list.

    Accepts plain repo id strings and objects with ``repo_id`` plus optional
    ``revision``, ``weights_file`` (alias ``weights``) and ``input_size``.
    """
    specs: List[SubmodelSpec] = []
    for entry in entries:
        if isinstance(entry, str):
            specs.append(SubmodelSpec(repo_id=entry))
        elif isinstance(entry, dict) and entry.get("repo_id"):
            specs.append(
                SubmodelSpec(
                    repo_id=entry["repo_id"],
                    revision=entry.get("revision") or None,
                    weights_file=entry.get("weights_file") or entry.get("weights") or None,
                    input_size=_normalize_input_size(entry.get("input_size")),
                )
            )
        else:
            raise ConfigurationError(
                message="Invalid submodel entry in fusion config",
                details={"entry": entry},
            )
    return specs


def load_local_manifest(path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load the local model manifest if present.

    Args:
        path: Explicit manifest path. When given and missing, an error is raised.
              When None, ``backend/model-manifest.json`` is used if it exists.
    """
    manifest_path = Path(path) if path else DEFAULT_MANIFEST_PATH
    if not manifest_path.exists():
        if path:
            raise ConfigurationError(
                message=f"Model manifest not found: {manifest_path}",
                details={"path": str(manifest_path)},
            )
        return None

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if not isinstance(manifest, dict):
        raise ConfigurationError(
            message="Model manifest must be a JSON object",
            details={"path": str(manifest_path)},
        )
    logger.info(f"Loaded model manifest from {manifest_path}")
    return manifest


def resolve_fusion_revision(
    fusion_repo_id: str,
    explicit_revision: Optional[str],
    manifest: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    Pick the fusion repo revision.

    Precedence: explicit setting (HF_FUSION_REVISION) > manifest > None (main).
    The manifest entry is only used when its repo_id matches the configured one.
    """
    if explicit_revision:
        return explicit_revision
    fusion = (manifest or {}).get("fusion") or {}
    if fusion.get("revision"):
        if fusion.get("repo_id") in (None, fusion_repo_id):
            return fusion["revision"]
        logger.warning(
            "Manifest fusion repo_id %s does not match configured %s; ignoring its revision",
            fusion.get("repo_id"),
            fusion_repo_id,
        )
    return None


def apply_manifest_overrides(
    specs: List[SubmodelSpec], manifest: Optional[Dict[str, Any]]
) -> List[SubmodelSpec]:
    """
    Override submodel revisions with those pinned in the local manifest.

    The fusion config still decides which submodels exist; manifest entries for
    repos not in the fusion config are ignored.
    """
    if not manifest:
        return specs
    pinned: Dict[str, str] = {}
    for entry in manifest.get("submodels") or []:
        if isinstance(entry, dict) and entry.get("repo_id") and entry.get("revision"):
            pinned[entry["repo_id"]] = entry["revision"]

    known = {spec.repo_id for spec in specs}
    for repo_id in pinned.keys() - known:
        logger.warning(f"Manifest pins {repo_id}, which the fusion config does not use; ignoring")

    result = []
    for spec in specs:
        if spec.repo_id in pinned:
            spec = SubmodelSpec(**{**asdict(spec), "revision": pinned[spec.repo_id]})
        result.append(spec)
    return result


def compute_manifest_hash(models: List[Dict[str, Any]]) -> str:
    """
    Stable hash of what is being served.

    Args:
        models: One dict per model with repo_id, revision, weights_file and
                input_size. Order matters (fusion first, then submodels in
                fusion order).
    """
    keys = ("repo_id", "revision", "weights_file", "input_size")
    canonical = [{k: m.get(k) for k in keys} for m in models]
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_probe_pins(
    probe_repo_id: str, manifest: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Pins for the foundation-model probe from the manifest's ``probe`` section::

        {"probe": {"repo_id": "...", "revision": "<sha or null>",
                   "backbones": {"<key>": {"repo_id": "...", "revision": "<sha>",
                                           "filename": "..."}}}}

    Returns ``{"revision": ..., "backbones": {...}}`` (empty when absent). The
    head revision is ignored when the manifest names a different repo; backbone
    pins apply regardless, since they describe upstream repos.
    """
    probe = (manifest or {}).get("probe") or {}
    if not isinstance(probe, dict):
        raise ConfigurationError(message="Manifest 'probe' must be an object")
    out: Dict[str, Any] = {"revision": None, "backbones": {}}
    if probe.get("revision"):
        if probe.get("repo_id") in (None, probe_repo_id):
            out["revision"] = probe["revision"]
        else:
            logger.warning(
                "Manifest probe repo_id %s does not match configured %s; ignoring its revision",
                probe.get("repo_id"),
                probe_repo_id,
            )
    for key, pin in (probe.get("backbones") or {}).items():
        if not isinstance(pin, dict):
            raise ConfigurationError(
                message="Manifest probe backbone pins must be objects", details={"key": key}
            )
        out["backbones"][key] = {
            k: pin.get(k) for k in ("repo_id", "revision", "filename", "timm_name") if pin.get(k)
        }
    return out
