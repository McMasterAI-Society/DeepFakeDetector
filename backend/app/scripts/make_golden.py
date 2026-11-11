"""
Generate golden per-model outputs for the regression test.

Loads the configured fusion model and submodels, runs every image in the
images directory through each submodel and the fusion model, and writes the
probabilities to a JSON file that ``tests/test_golden_outputs.py`` compares
against (tolerance 1e-3 by default).

Usage (from backend/):
    python -m app.scripts.make_golden                       # existing images
    python -m app.scripts.make_golden --synthetic 20        # create 20 fixed images first
    python -m app.scripts.make_golden --images-dir path/to/images --output out.json

Re-run it, and commit the result, whenever a served artifact changes on
purpose (new revision, weights file or input size).
"""

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

BACKEND_DIR = Path(__file__).resolve().parents[2]
GOLDEN_DIR = BACKEND_DIR / "tests" / "golden"
DEFAULT_IMAGES_DIR = GOLDEN_DIR / "images"
DEFAULT_OUTPUT = GOLDEN_DIR / "expected.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_TOLERANCE = 1e-3


def list_images(images_dir: Path) -> List[Path]:
    """Images in a directory, sorted by name."""
    if not images_dir.is_dir():
        return []
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def write_synthetic_images(images_dir: Path, count: int, seed: int = 1234) -> List[Path]:
    """
    Write deterministic synthetic PNGs (noise, gradients, checkerboards,
    non-square sizes). They are not meant to look real or fake; they pin
    preprocessing and weights so silent serving changes are caught.
    """
    import numpy as np
    from PIL import Image

    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    sizes = [(224, 224), (256, 256), (320, 240), (512, 384), (300, 500)]
    paths = []
    for i in range(count):
        w, h = sizes[i % len(sizes)]
        kind = i % 4
        if kind == 0:
            arr = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
        elif kind == 1:
            x = np.linspace(0, 255, w, dtype=np.float32)[None, :, None]
            y = np.linspace(0, 255, h, dtype=np.float32)[:, None, None]
            arr = np.concatenate([x + 0 * y, y + 0 * x, (x + y) / 2], axis=2).astype(np.uint8)
        elif kind == 2:
            block = 8 + 4 * (i % 5)
            yy, xx = np.mgrid[0:h, 0:w]
            board = (((yy // block) + (xx // block)) % 2 * 255).astype(np.uint8)
            arr = np.stack([board, 255 - board, board // 2], axis=2)
        else:
            base = rng.randint(0, 256, (h // 16 + 1, w // 16 + 1, 3), dtype=np.uint8)
            arr = np.asarray(
                Image.fromarray(base).resize((w, h), Image.BICUBIC), dtype=np.uint8
            )
        path = images_dir / f"synthetic_{i:02d}.png"
        Image.fromarray(arr).save(path, format="PNG")
        paths.append(path)
    return paths


def compute_outputs(registry: Any, image_path: Path) -> Dict[str, float]:
    """prob_fake per submodel name plus "fusion" for one image."""
    image_bytes = image_path.read_bytes()
    submodel_outputs = {}
    probs: Dict[str, float] = {}
    for name, wrapper in registry.get_all_submodels().items():
        output = wrapper.predict(image_bytes=image_bytes, explain=False)
        submodel_outputs[name] = output
        probs[name] = float(output["prob_fake"])
    fusion_output = registry.get_fusion().predict(submodel_outputs)
    probs["fusion"] = float(fusion_output["prob_fake"])
    return probs


def build_expected(registry: Any, images: List[Path], images_dir: Path) -> Dict[str, Any]:
    """Expected-output document for a loaded registry."""
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tolerance": DEFAULT_TOLERANCE,
        "manifest_hash": registry.get_manifest_hash(),
        "models": [
            {k: m.get(k) for k in ("name", "repo_id", "revision", "weights_file", "input_size")}
            for m in registry.list_models()
        ],
        "images": {
            str(p.relative_to(images_dir).as_posix()): compute_outputs(registry, p)
            for p in images
        },
    }


async def _load_registry(fusion_repo: str):
    from app.services.model_registry import ModelRegistry

    registry = ModelRegistry()
    await registry.load_from_fusion_repo(fusion_repo)
    return registry


def main() -> None:
    from app.core.config import settings
    from app.core.logging import setup_logging

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--synthetic", type=int, default=0,
                        help="Write N deterministic synthetic images into --images-dir first")
    parser.add_argument("--fusion-repo", default=settings.HF_FUSION_REPO_ID)
    args = parser.parse_args()

    setup_logging()
    if args.synthetic:
        write_synthetic_images(args.images_dir, args.synthetic)
    images = list_images(args.images_dir)
    if not images:
        raise SystemExit(f"No images found in {args.images_dir} (use --synthetic N to create some)")

    registry = asyncio.run(_load_registry(args.fusion_repo))
    expected = build_expected(registry, images, args.images_dir)
    # Image paths are stored relative to this directory, itself relative to the output file.
    expected["images_dir"] = os.path.relpath(args.images_dir.resolve(), args.output.resolve().parent).replace(os.sep, "/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(images)} golden entries to {args.output} (manifest_hash={expected['manifest_hash']})")


if __name__ == "__main__":
    main()
