"""
Golden-output regression test.

Compares per-model ``prob_fake`` (and the fusion output) for a fixed set of
images against ``tests/golden/expected.json``, within the tolerance stored in
that file (1e-3 by default). Generate or refresh the file with::

    python -m app.scripts.make_golden --synthetic 20

The test skips when the expected file or its images are missing, or when the
models cannot be loaded (no network / HF cache). Set ``GOLDEN_REQUIRED=1`` (CI)
to turn a model-load failure into a test failure instead of a skip.

Environment:
    GOLDEN_EXPECTED   path to the expected JSON (default tests/golden/expected.json)
    GOLDEN_REQUIRED   "1" to fail instead of skip when models cannot be loaded
"""

import asyncio
import json
import os
from pathlib import Path

import pytest

from app.scripts.make_golden import DEFAULT_OUTPUT, DEFAULT_TOLERANCE, compute_outputs

EXPECTED_PATH = Path(os.environ.get("GOLDEN_EXPECTED", DEFAULT_OUTPUT))


def _load_expected():
    if not EXPECTED_PATH.exists():
        pytest.skip(f"No golden expectations at {EXPECTED_PATH}; run app.scripts.make_golden")
    expected = json.loads(EXPECTED_PATH.read_text(encoding="utf-8"))
    images_dir = (EXPECTED_PATH.parent / expected.get("images_dir", "images")).resolve()
    images = {
        name: images_dir / name
        for name in expected.get("images", {})
        if (images_dir / name).exists()
    }
    if not images:
        pytest.skip(f"No golden images found in {images_dir}")
    return expected, images


@pytest.fixture(scope="module")
def golden():
    expected, images = _load_expected()
    try:
        from app.core.config import settings
        from app.services.model_registry import ModelRegistry

        registry = ModelRegistry()
        asyncio.run(registry.load_from_fusion_repo(settings.HF_FUSION_REPO_ID))
    except Exception as exc:  # models/weights unavailable
        if os.environ.get("GOLDEN_REQUIRED") == "1":
            raise
        pytest.skip(f"Models unavailable for golden test: {exc}")
    return expected, images, registry


def test_golden_prob_fake(golden):
    expected, images, registry = golden
    tolerance = float(expected.get("tolerance", DEFAULT_TOLERANCE))

    mismatches = []
    for name, path in sorted(images.items()):
        actual = compute_outputs(registry, path)
        for model, expected_prob in expected["images"][name].items():
            if model not in actual:
                mismatches.append(f"{name}: model {model} not loaded")
                continue
            diff = abs(actual[model] - float(expected_prob))
            if diff > tolerance:
                mismatches.append(
                    f"{name} [{model}]: expected {expected_prob:.6f}, got {actual[model]:.6f} (|diff|={diff:.2e})"
                )

    if mismatches:
        served = registry.get_manifest_hash()
        note = ""
        if served != expected.get("manifest_hash"):
            note = (
                f"\nServed manifest_hash {served} differs from golden "
                f"{expected.get('manifest_hash')}: a revision, weights file or input size changed."
            )
        pytest.fail(
            f"{len(mismatches)} golden mismatches (tolerance {tolerance}):\n"
            + "\n".join(mismatches[:50])
            + note
        )
