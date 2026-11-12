"""
Foundation-model probe: frozen CLIP-B/16 + DINOv3-B/16 features and a linear head.

Pipeline (must match training, see the foundation-probe experiment report):

1. Preprocess: EXIF transpose, RGB, short side to 512 (Lanczos), JPEG q90 round
   trip, short side to 224 (bicubic), center crop 224, per-backbone timm mean/std.
2. Per backbone: L2-normalised CLS token after blocks 6 and 9 (0-indexed layers
   5 and 8) and the L2-normalised head pre-logits pooled token. Features are
   concatenated in backbone order (2 x 3 x 768 = 4,608 dims for two ViT-B).
3. Head: StandardScaler + LogisticRegression, then Platt calibration
   ``p = sigmoid(platt_a * z + platt_b)``.
4. Verdict: ``ai`` if p >= upper band edge, ``real`` if p < lower band edge,
   ``inconclusive`` in between. ``pred``/``pred_int`` keep the binary decision at
   the validation threshold for backward compatibility.

Artifacts (one directory: the head repo snapshot or a local override):

    config.json   {"type": "foundation-probe", "backbones": [...], "timm_names": {...},
                   "layers": [5, 8], "threshold": t, "verdict_bands": [lo, hi],
                   "backbone_sources": {key: {"repo_id", "revision", "filename"}}}
    head.npz      scaler_mean, scaler_scale, coef, intercept, platt_a, platt_b,
                  threshold, format_version, head_version, feature_spec_hash
                  (portable numpy arrays, loaded with allow_pickle=False)

Backbone weights are resolved per backbone, first match wins:
    <PROBE_BACKBONE_DIR or artifact dir>/<key>.fp16.safetensors
    <PROBE_BACKBONE_DIR or artifact dir>/<key>.fp16.pt   (prototype format, weights_only)
    upstream timm repo on the Hub at the pinned revision (backbone_sources)
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from app.core.errors import ConfigurationError, InferenceError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseSubmodelWrapper

logger = get_logger(__name__)

PROBE_TYPE = "foundation-probe"
HEAD_FORMAT_VERSION = 1
SIZE = 224
CANON_SHORT_SIDE = 512
CANON_JPEG_QUALITY = 90
DEFAULT_LAYERS = (5, 8)

# Upstream timm repos at pinned commits (checked 2026-11-11). The head repo's
# config.json and the model manifest may override these.
DEFAULT_BACKBONE_SOURCES: Dict[str, Dict[str, str]] = {
    "clip_b16": {
        "timm_name": "vit_base_patch16_clip_224.openai",
        "repo_id": "timm/vit_base_patch16_clip_224.openai",
        "revision": "977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d",
        "filename": "pytorch_model.bin",
    },
    "dinov3_b16": {
        "timm_name": "vit_base_patch16_dinov3.lvd1689m",
        "repo_id": "timm/vit_base_patch16_dinov3.lvd1689m",
        "revision": "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da",
        "filename": "model.safetensors",
    },
}

# Display names of the per-backbone entries in /predict "submodels".
BACKBONE_DISPLAY_KEYS = {"clip_b16": "clip-b16", "dinov3_b16": "dinov3-b16"}

# Verdict band edges on the calibrated probability. Chosen on core-val-norm
# predictions (the selection split; nothing from the test sets):
#   lower 0.2: 3.0% of val fakes fall below it (fake called "real");
#   upper 0.7: 5.1% of val reals reach it (real called "ai");
#   the 0.311 decision threshold lies inside the band.
# Rationale: the known failure mode is high-quality real photos scoring as fake
# (16-30% FPR at 0.311 on DOCCI/Pexels/reddit reals), so the upper edge is set
# well above the threshold. Override with PROBE_VERDICT_BANDS or config.json.
DEFAULT_VERDICT_BANDS = (0.2, 0.7)

# The experiment calibrated logit(clip(predict_proba, 1e-6, 1 - 1e-6)), i.e. the
# LR logit clipped to +-log((1 - 1e-6) / 1e-6), before Platt scaling. A head.npz
# without "logit_clip" gets this value; export writes it explicitly.
DEFAULT_LOGIT_CLIP = float(np.log((1 - 1e-6) / 1e-6))

REQUIRED_HEAD_KEYS = (
    "scaler_mean", "scaler_scale", "coef", "intercept", "platt_a", "platt_b", "threshold",
)


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #

def canonicalize(im: Image.Image) -> Image.Image:
    """Training-time "norm" step: short side 512 (Lanczos) + JPEG q90 round trip."""
    w, h = im.size
    sc = CANON_SHORT_SIDE / min(w, h)
    im = im.resize((max(1, round(w * sc)), max(1, round(h * sc))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=CANON_JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def center_crop_view(im: Image.Image, size: int = SIZE) -> Image.Image:
    """Aspect-preserving short-side resize (bicubic) and center crop."""
    w, h = im.size
    s = size / min(w, h)
    nw, nh = max(size, round(w * s)), max(size, round(h * s))
    im = im.resize((nw, nh), Image.BICUBIC)
    left, top = (nw - size) // 2, (nh - size) // 2
    return im.crop((left, top, left + size, top + size))


def preprocess(im: Image.Image, canonical: bool = True) -> np.ndarray:
    """
    Full serving transform up to (not including) normalisation. Returns HxWx3 uint8.

    ``canonical=False`` skips the 512/JPEG-q90 step, for inputs that were already
    canonicalised offline (the experiment's -sq/-norm evaluation sets).
    """
    im = ImageOps.exif_transpose(im).convert("RGB")
    if canonical:
        im = canonicalize(im)
    return np.array(center_crop_view(im), dtype=np.uint8)


def oriented_size(im: Image.Image) -> Tuple[int, int]:
    """Image size after EXIF transpose (what the user sees)."""
    try:
        orientation = im.getexif().get(0x0112, 1)
    except Exception:  # noqa: BLE001
        orientation = 1
    w, h = im.size
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


# --------------------------------------------------------------------------- #
# Feature spec and head
# --------------------------------------------------------------------------- #

def feature_spec(config: Dict[str, Any]) -> Dict[str, Any]:
    """Canonical description of the features the head was fit on."""
    backbones = list(config.get("backbones") or [])
    timm_names = config.get("timm_names") or {}
    return {
        "backbones": [
            {"key": k, "timm_name": timm_names.get(k) or DEFAULT_BACKBONE_SOURCES.get(k, {}).get("timm_name")}
            for k in backbones
        ],
        "layers": [int(x) for x in config.get("layers", DEFAULT_LAYERS)],
        "token": "cls",
        "pooled": "head_pre_logits",
        "norm": "l2_per_layer",
        "preprocess": f"exif,rgb,short{CANON_SHORT_SIDE}_lanczos,jpeg{CANON_JPEG_QUALITY},"
                      f"short{SIZE}_bicubic,center_crop{SIZE},timm_mean_std",
    }


def feature_spec_hash(config: Dict[str, Any]) -> str:
    payload = json.dumps(feature_spec(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class ProbeHead:
    """StandardScaler + LogisticRegression + Platt, as plain arrays."""

    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    coef: np.ndarray
    intercept: float
    platt_a: float
    platt_b: float
    threshold: float
    head_version: str = "unversioned"
    feature_spec_hash: Optional[str] = None
    format_version: int = HEAD_FORMAT_VERSION
    logit_clip: Optional[float] = DEFAULT_LOGIT_CLIP

    @property
    def dim(self) -> int:
        return int(self.coef.shape[0])

    def raw_logit(self, feats: np.ndarray) -> np.ndarray:
        """LR decision function for (N, D) or (D,) features."""
        return ((feats - self.scaler_mean) / self.scaler_scale) @ self.coef + self.intercept

    def clip(self, z):
        """LR logit as seen by the Platt step (clipped like at calibration time)."""
        return np.clip(z, -self.logit_clip, self.logit_clip) if self.logit_clip else z

    def calibrated_logit(self, z):
        return self.platt_a * self.clip(z) + self.platt_b

    def calibrate(self, z):
        """Platt-calibrated probability of "fake" from the raw LR logit."""
        return 1.0 / (1.0 + np.exp(-self.calibrated_logit(z)))

    def to_npz_dict(self) -> Dict[str, np.ndarray]:
        return {
            "scaler_mean": np.asarray(self.scaler_mean, dtype=np.float64),
            "scaler_scale": np.asarray(self.scaler_scale, dtype=np.float64),
            "coef": np.asarray(self.coef, dtype=np.float64),
            "intercept": np.float64(self.intercept),
            "platt_a": np.float64(self.platt_a),
            "platt_b": np.float64(self.platt_b),
            "threshold": np.float64(self.threshold),
            "format_version": np.int64(self.format_version),
            "head_version": np.asarray(self.head_version),
            "feature_spec_hash": np.asarray(self.feature_spec_hash or ""),
            "logit_clip": np.float64(self.logit_clip or 0.0),
        }

    def save(self, path: Path) -> None:
        np.savez(path, **self.to_npz_dict())


def load_head(
    path: Path,
    expected_dim: Optional[int] = None,
    expected_spec_hash: Optional[str] = None,
) -> ProbeHead:
    """
    Load and validate a head.npz (no pickle).

    Raises ConfigurationError on missing keys, wrong shapes, non-finite values,
    an unknown format version or a feature-spec hash that does not match the
    config. A head without format metadata (the experiment's original export) is
    accepted with a warning.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(message=f"Probe head not found: {path}", details={"path": str(path)})
    try:
        with np.load(path, allow_pickle=False) as d:
            arrays = {k: d[k] for k in d.files}
    except Exception as e:  # noqa: BLE001
        raise ConfigurationError(message=f"Could not read probe head {path}: {e}", details={"path": str(path)})

    missing = [k for k in REQUIRED_HEAD_KEYS if k not in arrays]
    if missing:
        raise ConfigurationError(message=f"Probe head missing keys: {missing}", details={"path": str(path)})

    mean = arrays["scaler_mean"].astype(np.float64).ravel()
    scale = arrays["scaler_scale"].astype(np.float64).ravel()
    coef = arrays["coef"].astype(np.float64).ravel()
    if not (mean.shape == scale.shape == coef.shape):
        raise ConfigurationError(
            message="Probe head arrays have inconsistent shapes",
            details={"scaler_mean": mean.shape, "scaler_scale": scale.shape, "coef": coef.shape},
        )
    if expected_dim is not None and coef.shape[0] != expected_dim:
        raise ConfigurationError(
            message=f"Probe head expects {coef.shape[0]} features, backbones produce {expected_dim}",
            details={"path": str(path)},
        )
    scalars = {k: float(np.asarray(arrays[k]).reshape(())) for k in
               ("intercept", "platt_a", "platt_b", "threshold")}
    if not all(np.isfinite(a).all() for a in (mean, scale, coef)) or not all(
        math.isfinite(v) for v in scalars.values()
    ):
        raise ConfigurationError(message="Probe head contains non-finite values", details={"path": str(path)})
    if (scale <= 0).any():
        raise ConfigurationError(message="Probe head scaler_scale must be > 0", details={"path": str(path)})
    if not 0.0 < scalars["threshold"] < 1.0:
        raise ConfigurationError(message="Probe head threshold must be in (0, 1)", details={"path": str(path)})

    fmt = int(np.asarray(arrays["format_version"]).reshape(())) if "format_version" in arrays else None
    if fmt is not None and fmt > HEAD_FORMAT_VERSION:
        raise ConfigurationError(
            message=f"Probe head format_version {fmt} is newer than supported {HEAD_FORMAT_VERSION}",
            details={"path": str(path)},
        )
    head_version = str(arrays["head_version"]) if "head_version" in arrays else "unversioned"
    spec_hash = str(arrays["feature_spec_hash"]) if "feature_spec_hash" in arrays else ""
    # 0 means "no clipping"; absent means the experiment's convention.
    logit_clip = float(np.asarray(arrays["logit_clip"]).reshape(())) if "logit_clip" in arrays else DEFAULT_LOGIT_CLIP
    if not math.isfinite(logit_clip) or logit_clip < 0:
        raise ConfigurationError(message="Probe head logit_clip must be >= 0", details={"path": str(path)})
    if fmt is None:
        logger.warning(f"Probe head {path} has no format metadata; re-export it with app.scripts.export_probe_head")
    if spec_hash and expected_spec_hash and spec_hash != expected_spec_hash:
        raise ConfigurationError(
            message="Probe head was fit on a different feature spec than config.json describes",
            details={"head": spec_hash, "config": expected_spec_hash},
        )

    return ProbeHead(
        scaler_mean=mean, scaler_scale=scale, coef=coef,
        intercept=scalars["intercept"], platt_a=scalars["platt_a"], platt_b=scalars["platt_b"],
        threshold=scalars["threshold"], head_version=head_version,
        feature_spec_hash=spec_hash or None, format_version=fmt or 0,
        logit_clip=logit_clip or None,
    )


def verdict_for(prob_fake: float, bands: Sequence[float]) -> str:
    """Three-way verdict: "ai" at or above the upper edge, "real" below the lower edge."""
    lo, hi = bands
    if prob_fake >= hi:
        return "ai"
    if prob_fake < lo:
        return "real"
    return "inconclusive"


def parse_bands(value: Any) -> Optional[Tuple[float, float]]:
    """Parse "0.2,0.7" or [0.2, 0.7]; validate 0 <= lo <= hi <= 1."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [p for p in value.replace(";", ",").split(",") if p.strip()]
        value = [float(p) for p in parts]
    lo, hi = (float(v) for v in value)
    if not 0.0 <= lo <= hi <= 1.0:
        raise ConfigurationError(message=f"Invalid verdict bands {value}: need 0 <= lo <= hi <= 1")
    return lo, hi


# --------------------------------------------------------------------------- #
# Backbones
# --------------------------------------------------------------------------- #

class ProbeBackbone:
    """A frozen timm ViT with hooks for intermediate CLS tokens and patch tokens."""

    def __init__(self, key: str, model: torch.nn.Module, mean: Sequence[float], std: Sequence[float],
                 device: torch.device, dtype: torch.dtype, layers: Sequence[int] = DEFAULT_LAYERS,
                 source: str = "unknown"):
        self.key = key
        self.source = source
        self.layers = tuple(int(x) for x in layers)
        self.m = model.to(device=device, dtype=dtype).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor(mean, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device, dtype=dtype).view(1, 3, 1, 1)
        self.npt = int(getattr(model, "num_prefix_tokens", 0))
        self.has_cls = getattr(model, "cls_token", None) is not None
        self.embed_dim = int(getattr(model, "embed_dim", getattr(model, "num_features", 0)))
        self._buf: Dict[int, torch.Tensor] = {}
        self._capture_tokens = False
        self.tokens: Optional[torch.Tensor] = None
        n_blocks = len(self.m.blocks)
        if max(self.layers) >= n_blocks:
            raise ConfigurationError(message=f"{key}: layer {max(self.layers)} >= depth {n_blocks}")
        for li in self.layers:
            self.m.blocks[li].register_forward_hook(self._block_hook(li))
        self.m.patch_embed.register_forward_hook(self._patch_hook)

    @property
    def feature_dim(self) -> int:
        return self.embed_dim * (len(self.layers) + 1)

    def _block_hook(self, li: int):
        def hook(_mod, _inp, out):
            self._buf[li] = out[:, 0] if self.has_cls else out[:, self.npt:].mean(1)
        return hook

    def _patch_hook(self, _mod, _inp, out):
        if not self._capture_tokens:
            return None
        leaf = out.detach().requires_grad_(True)
        self.tokens = leaf
        return leaf

    def features(self, x01: torch.Tensor, capture_tokens: bool = False) -> torch.Tensor:
        """(B,3,H,W) in [0,1] -> (B, feature_dim) float32 L2-normalised features."""
        self._capture_tokens = capture_tokens
        try:
            x = (x01.to(self.dtype) - self.mean) / self.std
            pooled = self.m.forward_head(self.m.forward_features(x), pre_logits=True)
            feats = [self._buf[li] for li in self.layers] + [pooled]
            return torch.cat([F.normalize(f.float(), dim=-1) for f in feats], dim=-1)
        finally:
            self._buf.clear()
            self._capture_tokens = False


def _timm_create(timm_name: str, file: Optional[str] = None) -> torch.nn.Module:
    import timm
    if file is None:
        return timm.create_model(timm_name, pretrained=False, num_classes=0)
    # Same call that produced the training features (pretrained=True, num_classes=0),
    # pointed at a local file so timm's checkpoint filters still apply.
    return timm.create_model(
        timm_name, pretrained=True, num_classes=0,
        pretrained_cfg_overlay={"file": str(file), "source": "", "hf_hub_id": None, "url": None},
    )


def _norm_cfg(model: torch.nn.Module) -> Tuple[List[float], List[float]]:
    cfg = getattr(model, "pretrained_cfg", None) or {}
    return list(cfg.get("mean", (0.485, 0.456, 0.406))), list(cfg.get("std", (0.229, 0.224, 0.225)))


def resolve_backbone_module(
    key: str,
    source: Dict[str, Any],
    search_dirs: Sequence[Path],
    download: Callable[[str, str, Optional[str]], str],
) -> Tuple[torch.nn.Module, List[float], List[float], str]:
    """
    Build one backbone (fp32, CPU). Returns (module, mean, std, source description).

    ``download(repo_id, filename, revision)`` fetches the upstream file and returns
    its local path; it is only called when no local copy is found.
    """
    timm_name = source["timm_name"]
    for d in search_dirs:
        st = Path(d) / f"{key}.fp16.safetensors"
        if st.exists():
            model = _timm_create(timm_name, file=str(st))
            mean, std = _norm_cfg(model)
            return model, mean, std, f"local:{st}"
        pt = Path(d) / f"{key}.fp16.pt"
        if pt.exists():
            ck = torch.load(pt, map_location="cpu", weights_only=True)
            name = ck.get("name", timm_name)
            if name != timm_name:
                raise ConfigurationError(message=f"{pt} holds {name}, config expects {timm_name}")
            model = _timm_create(timm_name)
            model.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
            cfg = ck.get("cfg") or {}
            return model, list(cfg.get("mean", _norm_cfg(model)[0])), list(cfg.get("std", _norm_cfg(model)[1])), f"local:{pt}"
    repo_id, filename, revision = source.get("repo_id"), source.get("filename"), source.get("revision")
    if not repo_id or not filename:
        raise ConfigurationError(message=f"No weights for backbone {key}: no local file and no upstream repo")
    path = download(repo_id, filename, revision)
    model = _timm_create(timm_name, file=path)
    mean, std = _norm_cfg(model)
    return model, mean, std, f"hf:{repo_id}@{revision or 'main'}/{filename}"


def _default_download(repo_id: str, filename: str, revision: Optional[str]) -> str:
    from huggingface_hub import hf_hub_download
    from app.core.config import settings
    return hf_hub_download(repo_id=repo_id, filename=filename, revision=revision,
                           cache_dir=settings.HF_CACHE_DIR, token=settings.HF_TOKEN)


# --------------------------------------------------------------------------- #
# Wrapper
# --------------------------------------------------------------------------- #

class FoundationProbeWrapper(BaseSubmodelWrapper):
    """Frozen CLIP-B/16 + DINOv3-B/16 features with a calibrated linear head."""

    # Hooks for tests: replace to inject tiny backbones / avoid downloads.
    backbone_resolver: Callable[..., Tuple[torch.nn.Module, List[float], List[float], str]] = staticmethod(
        resolve_backbone_module
    )
    downloader: Callable[[str, str, Optional[str]], str] = staticmethod(_default_download)

    def __init__(self, repo_id: str, config: Dict[str, Any], local_path: str):
        super().__init__(repo_id, config, local_path)
        self.backbones: List[ProbeBackbone] = []
        self.head: Optional[ProbeHead] = None
        self.bands: Tuple[float, float] = DEFAULT_VERDICT_BANDS
        self.threshold: float = 0.5
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.backbone_sources: Dict[str, Dict[str, Any]] = {}

    # ---- loading -------------------------------------------------------- #

    def _settings(self):
        from app.core.config import settings
        return settings

    def _resolve_device(self) -> Tuple[torch.device, torch.dtype]:
        want = (self.config.get("device") or getattr(self._settings(), "PROBE_DEVICE", None) or "auto").lower()
        if want == "cpu" or (want == "auto" and not torch.cuda.is_available()):
            return torch.device("cpu"), torch.float32
        if not torch.cuda.is_available():
            logger.warning("PROBE_DEVICE=cuda but CUDA is unavailable; using CPU fp32")
            return torch.device("cpu"), torch.float32
        return torch.device("cuda"), torch.float16

    def _sources(self) -> Dict[str, Dict[str, Any]]:
        keys = list(self.config.get("backbones") or [])
        if not keys:
            raise ConfigurationError(message="foundation-probe config lists no backbones")
        timm_names = self.config.get("timm_names") or {}
        overrides = self.config.get("backbone_sources") or {}
        out = {}
        for k in keys:
            src = dict(DEFAULT_BACKBONE_SOURCES.get(k, {}))
            src.update({kk: v for kk, v in (overrides.get(k) or {}).items() if v is not None})
            if timm_names.get(k):
                src["timm_name"] = timm_names[k]
            if not src.get("timm_name"):
                raise ConfigurationError(message=f"No timm name for backbone {k}")
            out[k] = src
        return out

    def load(self) -> None:
        t0 = time.perf_counter()
        s = self._settings()
        self.device, self.dtype = self._resolve_device()
        self.backbone_sources = self._sources()
        layers = tuple(self.config.get("layers", DEFAULT_LAYERS))
        search_dirs = [Path(p) for p in (getattr(s, "PROBE_BACKBONE_DIR", None), self.local_path) if p]

        self.backbones = []
        for key, src in self.backbone_sources.items():
            bt0 = time.perf_counter()
            module, mean, std, where = type(self).backbone_resolver(key, src, search_dirs, type(self).downloader)
            bb = ProbeBackbone(key, module, mean, std, self.device, self.dtype, layers, source=where)
            self.backbones.append(bb)
            logger.info(f"probe backbone {key} from {where} on {self.device}/{self.dtype} "
                        f"in {time.perf_counter() - bt0:.2f}s")

        self.head = load_head(
            Path(self.local_path) / self.config.get("head_file", "head.npz"),
            expected_dim=sum(b.feature_dim for b in self.backbones),
            expected_spec_hash=feature_spec_hash(self.config),
        )
        # head.npz is the single source of truth for the threshold (a refit
        # changes it); config.json's copy is informational only.
        self.threshold = self.head.threshold
        cfg_thr = self.config.get("threshold")
        if cfg_thr is not None and abs(float(cfg_thr) - self.threshold) > 1e-9:
            logger.warning(f"config.json threshold {cfg_thr} differs from head.npz threshold "
                           f"{self.threshold}; using head.npz")
        self.bands = (parse_bands(getattr(s, "PROBE_VERDICT_BANDS", None))
                      or parse_bands(self.config.get("verdict_bands"))
                      or DEFAULT_VERDICT_BANDS)
        self.weights_file = self.config.get("head_file", "head.npz")
        self.input_size = SIZE
        self._predict_fn = self._forward
        logger.info(f"Loaded foundation probe {self.model_version} bands={self.bands} "
                    f"threshold={self.threshold:.4f} in {time.perf_counter() - t0:.2f}s")

    @property
    def model_version(self) -> str:
        head_v = self.head.head_version if self.head else "unloaded"
        spec = (self.head.feature_spec_hash if self.head and self.head.feature_spec_hash
                else feature_spec_hash(self.config))[:8]
        return f"{self.name}:{head_v}:{spec}"

    def backbone_slices(self) -> List[Tuple[str, slice]]:
        out, start = [], 0
        for bb in self.backbones:
            out.append((bb.key, slice(start, start + bb.feature_dim)))
            start += bb.feature_dim
        return out

    # ---- inference ------------------------------------------------------ #

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        # Same order as feature extraction: uint8 -> model dtype, then / 255.
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device).permute(2, 0, 1)[None].to(self.dtype).div_(255.0)

    def _forward(self, x01: torch.Tensor, explain: bool = False) -> Dict[str, Any]:
        head = self.head
        if not explain:
            with torch.inference_mode():
                feats = torch.cat([bb.features(x01) for bb in self.backbones], dim=-1)
            return {"feats": feats.cpu().numpy().astype(np.float64)[0]}

        # Explanation: gradient x activation of the raw LR logit at the patch-token
        # embeddings of every backbone (one forward + one backward).
        with torch.enable_grad():
            feats = torch.cat([bb.features(x01, capture_tokens=True) for bb in self.backbones], dim=-1)
            w = torch.as_tensor(head.coef / head.scaler_scale, device=feats.device, dtype=torch.float32)
            # z - intercept + const: the constant does not change gradients.
            z_lin = (feats[0] * w).sum()
            scale = 256.0 if self.dtype == torch.float16 else 1.0  # avoid fp16 grad underflow
            (z_lin * scale).backward()
        maps = {}
        for bb in self.backbones:
            tok, g = bb.tokens, bb.tokens.grad
            rel = (tok.float() * g.float()).sum(-1)[0] / scale  # (N,) or (H, W)
            if rel.dim() == 1:
                side = int(round(math.sqrt(rel.shape[0])))
                rel = rel[: side * side].view(side, side)
            maps[bb.key] = rel.detach().cpu().numpy().astype(np.float64)
            bb.tokens = None
        return {"feats": feats.detach().cpu().numpy().astype(np.float64)[0], "relevance": maps}

    def predict(self, image: Optional[Image.Image] = None, image_bytes: Optional[bytes] = None,
                explain: bool = False, canonicalize: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Predict one image. ``canonicalize=False`` is only for parity checks on
        images that were canonicalised offline; serving always canonicalises.
        """
        if not self.backbones or self.head is None:
            raise InferenceError(message="Model not loaded", details={"repo_id": self.repo_id})
        if image is None:
            if image_bytes is None:
                raise InferenceError(message="No image provided", details={"repo_id": self.repo_id})
            image = Image.open(io.BytesIO(image_bytes))
        try:
            orig_size = oriented_size(image)
            x = self._to_tensor(preprocess(image, canonical=canonicalize))
            out = self._forward(x, explain=explain)
            return self._build_output(out, orig_size, explain)
        except (InferenceError, ConfigurationError):
            raise
        except Exception as e:  # noqa: BLE001
            logger.error(f"Prediction failed for {self.repo_id}: {e}")
            raise InferenceError(message=f"Prediction failed: {e}", details={"repo_id": self.repo_id})

    def _build_output(self, out: Dict[str, Any], orig_size: Tuple[int, int], explain: bool) -> Dict[str, Any]:
        head = self.head
        f = out["feats"]
        std = (f - head.scaler_mean) / head.scaler_scale
        z = float(std @ head.coef + head.intercept)
        cal = float(head.calibrated_logit(np.float64(z)))
        p = float(1.0 / (1.0 + math.exp(-cal)))
        pred_int = int(p >= self.threshold)

        # Per-backbone share of the LR logit: dot product of each backbone's
        # standardised features with its slice of the coefficients. They sum,
        # with the intercept, to the raw logit.
        slices = dict(self.backbone_slices())
        contrib = {k: float(std[sl] @ head.coef[sl]) for k, sl in slices.items()}
        total_abs = sum(abs(v) for v in contrib.values()) or 1.0
        backbones: Dict[str, Dict[str, Any]] = {}
        for k, c in contrib.items():
            # Counterfactual probability with the other backbones at their
            # training-mean features (contribution 0).
            p_k = float(head.calibrate(np.float64(c + head.intercept)))
            backbones[k] = {
                "prob_fake": p_k,
                "pred_int": int(p_k >= self.threshold),
                "pred": "fake" if p_k >= self.threshold else "real",
                "logit_contribution": c,
                "contribution_percentage": 100.0 * abs(c) / total_abs,
                "coef_norm": float(np.linalg.norm(head.coef[slices[k]])),
            }

        result: Dict[str, Any] = {
            "pred_int": pred_int,
            "pred": "fake" if pred_int else "real",
            "prob_fake": p,
            "verdict": verdict_for(p, self.bands),
            "model_version": self.model_version,
            "backbones": backbones,
            "meta": {
                "model": self.name,
                "threshold": self.threshold,
                "verdict_bands": list(self.bands),
                "raw_logit": z,
                "calibrated_logit": cal,
                "intercept": head.intercept,
                "logits": [-cal / 2, cal / 2],
            },
        }

        if explain and "relevance" in out:
            from app.services.explainability import compute_focus_summary, heatmap_to_base64
            sign = 1.0 if pred_int else -1.0
            rel = out["relevance"]
            combined = sum(rel.values())
            for k, r in rel.items():
                hm = relevance_to_heatmap(r, sign)
                backbones[k]["heatmap_base64"] = heatmap_to_base64(place_on_canvas(hm, orig_size))
                backbones[k]["explainability_type"] = "input_gradient"
                backbones[k]["focus_summary"] = compute_focus_summary(hm)
            hm = relevance_to_heatmap(combined, sign)
            result["heatmap_base64"] = heatmap_to_base64(place_on_canvas(hm, orig_size))
            result["explainability_type"] = "input_gradient"
            result["focus_summary"] = compute_focus_summary(hm)
        return result


# --------------------------------------------------------------------------- #
# Heatmap helpers
# --------------------------------------------------------------------------- #

def relevance_to_heatmap(rel: np.ndarray, sign: float, size: int = SIZE) -> np.ndarray:
    """
    Token relevance grid -> [0, 1] heatmap of the evidence for the decision.

    Keeps tokens whose gradient x activation points towards the predicted class
    (sign=+1 for fake, -1 for real); falls back to magnitude when none do.
    Normalised by the 99th percentile so one token does not wash out the map.
    """
    r = np.clip(sign * np.asarray(rel, dtype=np.float64), 0, None)
    if not np.any(r > 0):
        r = np.abs(np.asarray(rel, dtype=np.float64))
    img = Image.fromarray(r.astype(np.float32), mode="F").resize((size, size), Image.BICUBIC)
    hm = np.clip(np.asarray(img, dtype=np.float64), 0, None)
    top = np.percentile(hm, 99) if np.any(hm > 0) else 1.0
    return np.clip(hm / (top or 1.0), 0.0, 1.0)


def place_on_canvas(hm: np.ndarray, orig_size: Tuple[int, int], short: int = SIZE) -> np.ndarray:
    """
    Put the center-crop heatmap back into the frame of the original image.

    The frontend overlays heatmaps on the whole uploaded image, but the model
    only sees the center square; regions outside it get 0 (not analysed).
    """
    w, h = orig_size
    s = short / min(w, h)
    cw, ch = max(short, round(w * s)), max(short, round(h * s))
    canvas = np.zeros((ch, cw), dtype=np.float64)
    left, top = (cw - short) // 2, (ch - short) // 2
    canvas[top:top + short, left:left + short] = hm
    return canvas
