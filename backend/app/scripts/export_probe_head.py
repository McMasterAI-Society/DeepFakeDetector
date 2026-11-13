"""
Export a foundation-probe head to the portable serving format.

Reads the experiment's sklearn head (head.joblib: StandardScaler + LogisticRegression
pipeline, Platt [a, b], threshold, config) or an existing head.npz, and writes a
directory ready to upload to the head repo (DeepFakeDetector/fm-probe-clip-dinov3):

    config.json        serving config (backbones, pinned upstream revisions,
                       layers, threshold, verdict bands, head version)
    head.npz           scaler_mean, scaler_scale, coef, intercept, platt_a,
                       platt_b, threshold, logit_clip, format_version,
                       head_version, feature_spec_hash (no pickle)
    README.md          model card (from backend/hf/fm-probe-clip-dinov3/MODEL_CARD.md)
    LICENSE-DINOv3.md  DINOv3 licence text (must ship with the model)

Usage (from backend/):
    python -m app.scripts.export_probe_head \\
        --joblib <S>/experiments/foundation/results/<name>/head.joblib \\
        --val-preds <S>/experiments/foundation/results/<name>/core-val-norm__none.preds.csv \\
        --head-version <name>-<yyyymmdd> --out <dir>

Options:
    --val-preds CSV        derive verdict bands from validation predictions
                           (columns label, prob_fake): lower edge = 3rd percentile
                           of fake scores, upper edge = 95th percentile of real
                           scores (rounded to 0.01). Without it the template's
                           bands are kept; they must contain the threshold.
    --verdict-bands lo,hi  set the bands explicitly (overrides --val-preds).
    --backbone-dir DIR     also write <key>.fp16.safetensors from <key>.fp16.pt in
                           DIR (optional; lets the head repo serve fp16 backbone
                           copies instead of the upstream fp32 checkpoints).

head.joblib is a pickle: only convert files you produced yourself. The serving
side never unpickles anything.
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.models.wrappers.foundation_probe_wrapper import (
    DEFAULT_BACKBONE_SOURCES,
    DEFAULT_LOGIT_CLIP,
    HEAD_FORMAT_VERSION,
    PROBE_TYPE,
    ProbeHead,
    feature_spec_hash,
    load_head,
    parse_bands,
)

BACKEND_DIR = Path(__file__).resolve().parents[2]
TEMPLATE_DIR = BACKEND_DIR / "hf" / "fm-probe-clip-dinov3"
LAYER_SETS = {"multi": [5, 8]}
FNR_QUANTILE = 0.03  # share of val fakes allowed below the lower edge
FPR_QUANTILE = 0.05  # share of val reals allowed at/above the upper edge


def head_from_joblib(path: Path) -> Tuple[ProbeHead, Dict[str, Any]]:
    """ProbeHead and training config from the experiment's head.joblib."""
    import joblib  # noqa: PLC0415  (offline conversion only)

    d = joblib.load(path)
    pipe = d["head"]
    scaler, lr = pipe.steps[0][1], pipe.steps[-1][1]
    if lr.coef_.shape[0] != 1:
        raise SystemExit("expected a binary LogisticRegression")
    a, b = d["calib"]
    head = ProbeHead(
        scaler_mean=np.asarray(scaler.mean_, dtype=np.float64),
        scaler_scale=np.asarray(scaler.scale_, dtype=np.float64),
        coef=np.asarray(lr.coef_[0], dtype=np.float64),
        intercept=float(lr.intercept_[0]),
        platt_a=float(a),
        platt_b=float(b),
        threshold=float(d["threshold"]),
        logit_clip=DEFAULT_LOGIT_CLIP,
    )
    return head, dict(d.get("config") or {}), pipe


def check_against_pipeline(head: ProbeHead, pipe, n: int = 64, seed: int = 0) -> float:
    """Max |p_numpy - p_sklearn| on random unit-norm-ish features."""
    rng = np.random.default_rng(seed)
    X = head.scaler_mean + rng.standard_normal((n, head.dim)) * head.scaler_scale
    p_np = head.calibrate(head.raw_logit(X))
    p_lr = np.clip(pipe.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    z_lr = np.log(p_lr / (1 - p_lr))
    p_sk = 1 / (1 + np.exp(-(head.platt_a * z_lr + head.platt_b)))
    return float(np.max(np.abs(p_np - p_sk)))


def bands_from_val(path: Path) -> Tuple[float, float]:
    """Verdict bands from validation predictions (see module docstring)."""
    labels, probs = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append(int(row["label"]))
            probs.append(float(row["prob_fake"]))
    labels, probs = np.asarray(labels), np.asarray(probs)
    lo = float(np.quantile(probs[labels == 1], FNR_QUANTILE))
    hi = float(np.quantile(probs[labels == 0], 1 - FPR_QUANTILE))
    return round(lo, 2), round(hi, 2)


def band_stats(path: Path, bands: Tuple[float, float]) -> Dict[str, float]:
    labels, probs = [], []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            labels.append(int(row["label"]))
            probs.append(float(row["prob_fake"]))
    labels, probs = np.asarray(labels), np.asarray(probs)
    lo, hi = bands
    r, f = probs[labels == 0], probs[labels == 1]
    return {
        "real_called_ai": float((r >= hi).mean()),
        "real_inconclusive": float(((r >= lo) & (r < hi)).mean()),
        "fake_called_real": float((f < lo).mean()),
        "fake_inconclusive": float(((f >= lo) & (f < hi)).mean()),
    }


def export_backbones(src_dir: Path, out: Path, keys) -> None:
    import torch
    from safetensors.torch import save_file

    for key in keys:
        ck = torch.load(src_dir / f"{key}.fp16.pt", map_location="cpu", weights_only=True)
        sd = {k: v.half().contiguous() for k, v in ck["state_dict"].items()}
        save_file(sd, str(out / f"{key}.fp16.safetensors"), metadata={"timm_name": ck["name"]})
        print(f"wrote {key}.fp16.safetensors")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--joblib", type=Path, help="experiment head.joblib (sklearn pipeline)")
    src.add_argument("--npz", type=Path, help="existing head.npz (e.g. best/head.npz) to re-export with metadata")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--head-version", required=True, help="e.g. clipdino_both_coco-20261111")
    ap.add_argument("--template", type=Path, default=TEMPLATE_DIR, help="dir with config.json, MODEL_CARD.md, LICENSE-DINOv3.md")
    ap.add_argument("--val-preds", type=Path, help="core-val-norm preds CSV for the verdict bands")
    ap.add_argument("--verdict-bands", help="explicit 'lo,hi'")
    ap.add_argument("--backbone-dir", type=Path, help="dir with <key>.fp16.pt to convert to safetensors")
    a = ap.parse_args(argv)

    config = json.loads((a.template / "config.json").read_text(encoding="utf-8"))
    if config.get("type") != PROBE_TYPE:
        raise SystemExit(f"template config is not type {PROBE_TYPE}")

    if a.joblib:
        head, train_cfg, pipe = head_from_joblib(a.joblib)
        bbs = (train_cfg.get("bbs") or "").split("+")
        if bbs != [""] and bbs != config["backbones"]:
            raise SystemExit(f"head backbones {bbs} != template {config['backbones']}")
        layers = train_cfg.get("layers")
        if layers and LAYER_SETS.get(layers) != config["layers"]:
            raise SystemExit(f"head layer set {layers!r} does not match template layers {config['layers']}")
        diff = check_against_pipeline(head, pipe)
        print(f"numpy head vs sklearn pipeline: max |dp| = {diff:.2e}")
        if diff > 1e-9:
            raise SystemExit("exported head does not reproduce the sklearn pipeline")
        config["train_config"] = train_cfg
    else:
        head = load_head(a.npz)

    head.head_version = a.head_version
    head.format_version = HEAD_FORMAT_VERSION
    head.feature_spec_hash = feature_spec_hash(config)

    bands = parse_bands(a.verdict_bands) or (bands_from_val(a.val_preds) if a.val_preds else None) \
        or parse_bands(config.get("verdict_bands"))
    if bands is None:
        raise SystemExit("no verdict bands: pass --val-preds or --verdict-bands")
    if not bands[0] <= head.threshold <= bands[1]:
        raise SystemExit(f"threshold {head.threshold:.4f} lies outside verdict bands {bands}; "
                         f"pass --val-preds or --verdict-bands")

    config["threshold"] = head.threshold
    config["verdict_bands"] = list(bands)
    config["head_version"] = a.head_version
    config["feature_spec_hash"] = head.feature_spec_hash
    sources = config.setdefault("backbone_sources", {})
    for key in config["backbones"]:
        sources.setdefault(key, {k: v for k, v in DEFAULT_BACKBONE_SOURCES.get(key, {}).items() if k != "timm_name"})

    a.out.mkdir(parents=True, exist_ok=True)
    head.save(a.out / "head.npz")
    reloaded = load_head(a.out / "head.npz", expected_dim=head.dim, expected_spec_hash=head.feature_spec_hash)
    assert np.array_equal(reloaded.coef, head.coef) and reloaded.threshold == head.threshold
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(a.template / "MODEL_CARD.md", a.out / "README.md")
    shutil.copyfile(a.template / "LICENSE-DINOv3.md", a.out / "LICENSE-DINOv3.md")
    if a.backbone_dir:
        export_backbones(a.backbone_dir, a.out, config["backbones"])

    print(f"wrote {a.out}: head_version={a.head_version} dim={head.dim} threshold={head.threshold:.4f} "
          f"bands={list(bands)} spec={head.feature_spec_hash[:12]}")
    if a.val_preds:
        print("val band stats:", json.dumps(band_stats(a.val_preds, bands)))
    print("Update MODEL_CARD metrics if the head changed; test with PROBE_LOCAL_PATH=<out> before upload.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
