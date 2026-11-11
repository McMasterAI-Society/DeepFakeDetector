"""
Compare explainability methods on a folder of images.

For every model and method it reports deletion AUC (lower is better),
insertion AUC (higher is better), the same metrics for a random-map control,
explain latency, and, on a few images, a parameter-randomisation sanity
check (Spearman correlation with the map of a randomised model; lower is
better).

Models are loaded from local directories that contain the Hugging Face repo
files (config.json + weights), e.g. a hub cache. Nothing is downloaded.

Usage:
    python -m app.scripts.explain_benchmark \\
        --images path/to/images \\
        --models-root path/to/.hf_cache \\
        --out explain_benchmark.json

    # or name model directories explicitly
    python -m app.scripts.explain_benchmark --images imgs \\
        --model-dir path/to/cnn-transfer-final --model-dir path/to/vit-base-final
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.services import explain_metrics as em
from app.services.explainability import (
    compute_cam,
    compute_transformer_attribution,
    predicted_class,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# A method is (name, fn(model, x, target_class) -> heatmap [H, W] >= 0)
Method = Tuple[str, Callable[[nn.Module, torch.Tensor, int], np.ndarray]]


# ---------------------------------------------------------------------------
# Per-architecture adapters
# ---------------------------------------------------------------------------

class Adapter:
    """Glue between a loaded wrapper and the generic metrics."""

    kind = "generic"

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.model: nn.Module = wrapper._model
        self.device = next(self.model.parameters()).device
        self.threshold = float(getattr(wrapper, "_threshold", 0.5))

    def prepare(self, image: Image.Image) -> torch.Tensor:
        return self.wrapper._transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def logits(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        out = model(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    def class_prob(self, model: nn.Module, x: torch.Tensor, target: int) -> torch.Tensor:
        lg = self.logits(model, x)
        if lg.dim() == 1 or lg.shape[-1] == 1:
            p = torch.sigmoid(lg.reshape(-1))
            return p if target == 1 else 1 - p
        return torch.softmax(lg, dim=-1)[:, target]

    def target(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            lg = self.logits(self.model, x)
        if lg.dim() == 1:
            lg = lg.unsqueeze(-1)
        return predicted_class(lg, self.threshold)

    def methods(self) -> List[Method]:
        raise NotImplementedError

    def cascade(self) -> Optional[List[str]]:
        return None


class CNNTransferAdapter(Adapter):
    kind = "cnn_transfer"

    def methods(self) -> List[Method]:
        def cam(method: str, layer: int):
            def fn(model, x, target):
                return compute_cam(model, x, model.features[layer], method=method, target_class=target).heatmap
            return fn
        return [
            ("hirescam@features[5]", cam("hirescam", 5)),
            ("gradcam++@features[5]", cam("gradcam++", 5)),
            ("gradcam@features[5]", cam("gradcam", 5)),
            ("hirescam@features[-1]", cam("hirescam", -1)),
            ("gradcam@features[-1] (previous default)", cam("gradcam", -1)),
        ]

    def cascade(self):
        return ["classifier", "features.8", "features.7", "features.6", "features.5"]


class TransformerAdapter(Adapter):
    kind = "transformer"

    def __init__(self, wrapper):
        super().__init__(wrapper)
        self.settings = wrapper._attribution_settings()

    def methods(self) -> List[Method]:
        def attr(method: str):
            def fn(model, x, target):
                s = dict(self.settings)
                s.pop("method")
                # Blocks must come from the model being explained (it may be a
                # randomised copy), so look them up by position.
                s["blocks"] = _find_blocks(model, len(s["blocks"]))
                return compute_transformer_attribution(model, x, method=method, target_class=target, **s).heatmap
            return fn
        return [
            ("grad_rollout", attr("grad_rollout")),
            ("rollout (previous default)", attr("rollout")),
        ]


class GradfieldAdapter(Adapter):
    kind = "gradfield"

    def prepare(self, image: Image.Image) -> torch.Tensor:
        img = self.wrapper._resize(image.convert("RGB"))
        lum = self.wrapper._rgb_to_luminance(transforms.functional.to_tensor(img))
        return lum.unsqueeze(0).to(self.device)

    def methods(self) -> List[Method]:
        def last_relu(model):
            return [m for m in model.cnn if isinstance(m, nn.ReLU)][-1]

        def signed_toward_target(model, x, target):
            # Positive part of the signed map for the explained class
            r = compute_cam(model, x, last_relu(model), method="hirescam", target_class=target)
            return r.heatmap

        def magnitude(model, x, target):
            r = compute_cam(model, x, last_relu(model), method="hirescam", target_class=1)
            return np.abs(r.signed_map)

        def legacy_abs(model, x, target):
            # Previous behaviour: Grad-CAM at the last Conv2d, abs() of the map
            conv = [m for m in model.cnn if isinstance(m, nn.Conv2d)][-1]
            r = compute_cam(model, x, conv, method="gradcam", target_class=1)
            return np.abs(r.signed_map)

        return [
            ("hirescam_signed (positive, toward predicted class)", signed_toward_target),
            ("hirescam_signed |magnitude| (UI heatmap)", magnitude),
            ("gradcam |abs| @last conv (previous default)", legacy_abs),
        ]

    def cascade(self):
        return ["classifier", "embedding", "cnn"]


def _find_blocks(model: nn.Module, n: int):
    for name in ("vit.blocks", "blocks"):
        obj = model
        try:
            for part in name.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if len(obj) == n:
            return list(obj)
    raise AttributeError("transformer blocks not found")


def make_adapter(wrapper) -> Adapter:
    cls = type(wrapper).__name__
    if cls == "CNNTransferWrapper":
        return CNNTransferAdapter(wrapper)
    if cls in ("ViTBaseWrapper", "DeiTDistilledWrapper"):
        return TransformerAdapter(wrapper)
    if cls == "GradfieldCNNWrapper":
        return GradfieldAdapter(wrapper)
    raise ValueError(f"No benchmark adapter for {cls}")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_wrapper(model_dir: Path):
    from app.services.model_registry import get_wrapper_class

    config = json.loads((model_dir / "config.json").read_text())
    if "submodel_order" in config:
        return None  # fusion repo
    wrapper_cls = get_wrapper_class(config)
    repo_id = model_dir.name.replace("--", "/")
    wrapper = wrapper_cls(repo_id=repo_id, config=config, local_path=str(model_dir))
    wrapper.load()
    return wrapper


def discover_model_dirs(root: Path, suffix: Optional[str]) -> List[Path]:
    dirs = [p for p in sorted(root.iterdir()) if (p / "config.json").exists()]
    if suffix:
        dirs = [p for p in dirs if p.name.endswith(suffix)]
    return dirs


def list_images(folder: Path, limit: Optional[int]) -> List[Path]:
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _mean(xs: List[float]) -> Optional[float]:
    return float(np.mean(xs)) if xs else None


def benchmark_model(
    adapter: Adapter,
    images: List[Path],
    steps: int,
    baseline: str,
    sanity_images: int,
) -> Dict[str, Any]:
    model = adapter.model
    methods = adapter.methods()
    per_method: Dict[str, Dict[str, List[float]]] = {
        name: {"deletion_auc": [], "insertion_auc": [], "latency_ms": [], "sanity_spearman": []}
        for name, _ in methods
    }
    control = {"deletion_auc": [], "insertion_auc": []}

    for i, path in enumerate(images):
        x = adapter.prepare(Image.open(path))
        target = adapter.target(x)

        def predict_fn(batch, _t=target):
            return adapter.class_prob(model, batch, _t)

        hw = tuple(x.shape[-2:])
        rnd = em.deletion_insertion(predict_fn, x, em.random_heatmap(hw, seed=i), steps=steps, baseline=baseline)
        control["deletion_auc"].append(rnd["deletion_auc"])
        control["insertion_auc"].append(rnd["insertion_auc"])

        for name, fn in methods:
            if adapter.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            heatmap = fn(model, x, target)
            if adapter.device.type == "cuda":
                torch.cuda.synchronize()
            per_method[name]["latency_ms"].append((time.perf_counter() - t0) * 1000)

            r = em.deletion_insertion(predict_fn, x, heatmap, steps=steps, baseline=baseline)
            per_method[name]["deletion_auc"].append(r["deletion_auc"])
            per_method[name]["insertion_auc"].append(r["insertion_auc"])

            if i < sanity_images:
                s = em.parameter_randomization_check(
                    lambda m, xx, _fn=fn, _t=target: _fn(m, xx, _t), model, x,
                )
                per_method[name]["sanity_spearman"].append(s["final_spearman"])
        print(f"  [{i + 1}/{len(images)}] {path.name}", file=sys.stderr)

    summary = {
        name: {
            "deletion_auc": _mean(v["deletion_auc"]),
            "insertion_auc": _mean(v["insertion_auc"]),
            "latency_ms": _mean(v["latency_ms"]),
            "sanity_spearman": _mean(v["sanity_spearman"]),
            "n_images": len(v["deletion_auc"]),
        }
        for name, v in per_method.items()
    }
    best = min(summary, key=lambda k: summary[k]["deletion_auc"]) if summary else None
    return {
        "kind": adapter.kind,
        "methods": summary,
        "random_control": {k: _mean(v) for k, v in control.items()},
        "best_by_deletion_auc": best,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", required=True, type=Path, help="Folder of images (searched recursively)")
    parser.add_argument("--models-root", type=Path, help="Folder whose subfolders are model repos")
    parser.add_argument("--suffix", default="-final", help="Only use --models-root subfolders ending in this ('' for all)")
    parser.add_argument("--model-dir", type=Path, action="append", default=[], help="Model repo folder (repeatable)")
    parser.add_argument("--limit", type=int, default=None, help="Max images")
    parser.add_argument("--steps", type=int, default=20, help="Deletion/insertion steps")
    parser.add_argument("--baseline", default="blur", choices=["blur", "zero", "mean"])
    parser.add_argument("--sanity-images", type=int, default=2, help="Images used for the randomisation check")
    parser.add_argument("--out", type=Path, default=Path("explain_benchmark.json"))
    args = parser.parse_args(argv)

    dirs = list(args.model_dir)
    if args.models_root:
        dirs += discover_model_dirs(args.models_root, args.suffix or None)
    if not dirs:
        parser.error("give --models-root or --model-dir")
    images = list_images(args.images, args.limit)
    if not images:
        parser.error(f"no images found in {args.images}")

    results: Dict[str, Any] = {
        "config": {
            "images": str(args.images), "n_images": len(images), "steps": args.steps,
            "baseline": args.baseline, "sanity_images": args.sanity_images,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "notes": "deletion_auc: lower is better; insertion_auc and lower sanity_spearman: better",
        },
        "models": {},
    }
    for d in dirs:
        wrapper = load_wrapper(d)
        if wrapper is None:
            continue
        print(f"{wrapper.name}: {len(images)} images", file=sys.stderr)
        results["models"][wrapper.name] = benchmark_model(
            make_adapter(wrapper), images, args.steps, args.baseline, args.sanity_images,
        )
        args.out.write_text(json.dumps(results, indent=2))

    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps({m: r["best_by_deletion_auc"] for m, r in results["models"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
