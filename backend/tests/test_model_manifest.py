"""
Unit tests for model manifest parsing, revision-pinned caching and gradfield
artifact resolution. No network access and no real weights are used.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.errors import ConfigurationError
from app.services import model_manifest
from app.services.hf_hub_service import HFHubService
from app.services.model_manifest import (
    SubmodelSpec,
    apply_manifest_overrides,
    compute_manifest_hash,
    load_local_manifest,
    parse_submodel_entries,
    resolve_fusion_revision,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


# ---------------------------------------------------------------------------
# Fusion config parsing
# ---------------------------------------------------------------------------

class TestParseSubmodelEntries:
    def test_plain_strings(self):
        specs = parse_submodel_entries(["Org/a", "Org/b"])
        assert specs == [SubmodelSpec(repo_id="Org/a"), SubmodelSpec(repo_id="Org/b")]

    def test_objects(self):
        specs = parse_submodel_entries([
            {"repo_id": "Org/a", "revision": SHA_A, "weights_file": "w.pth", "input_size": 224},
        ])
        assert specs == [SubmodelSpec("Org/a", SHA_A, "w.pth", 224)]

    def test_mixed_and_aliases(self):
        specs = parse_submodel_entries([
            "Org/a",
            {"repo_id": "Org/b", "weights": "v2.pth", "input_size": [256, 256]},
        ])
        assert specs[0] == SubmodelSpec(repo_id="Org/a")
        assert specs[1] == SubmodelSpec("Org/b", None, "v2.pth", 256)

    @pytest.mark.parametrize("entry", [{"revision": SHA_A}, 42, None])
    def test_invalid_entry(self, entry):
        with pytest.raises(ConfigurationError):
            parse_submodel_entries([entry])


# ---------------------------------------------------------------------------
# Local manifest
# ---------------------------------------------------------------------------

class TestLocalManifest:
    def test_missing_default_returns_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(model_manifest, "DEFAULT_MANIFEST_PATH", tmp_path / "nope.json")
        assert load_local_manifest() is None

    def test_missing_explicit_path_raises(self, tmp_path):
        with pytest.raises(ConfigurationError):
            load_local_manifest(str(tmp_path / "nope.json"))

    def test_loads_default(self, tmp_path, monkeypatch):
        path = tmp_path / "model-manifest.json"
        path.write_text(json.dumps({"fusion": {"repo_id": "Org/f", "revision": SHA_A}}))
        monkeypatch.setattr(model_manifest, "DEFAULT_MANIFEST_PATH", path)
        assert load_local_manifest()["fusion"]["revision"] == SHA_A

    def test_submodel_revision_override(self):
        specs = [
            SubmodelSpec("Org/a", revision="old", weights_file="w.pth"),
            SubmodelSpec("Org/b"),
        ]
        manifest = {"submodels": [
            {"repo_id": "Org/a", "revision": SHA_A},
            {"repo_id": "Org/unused", "revision": SHA_B},
        ]}
        result = apply_manifest_overrides(specs, manifest)
        assert result[0] == SubmodelSpec("Org/a", SHA_A, "w.pth", None)
        assert result[1] == SubmodelSpec("Org/b")
        assert [s.repo_id for s in result] == ["Org/a", "Org/b"]
        # Inputs are not mutated
        assert specs[0].revision == "old"

    def test_no_manifest_is_identity(self):
        specs = [SubmodelSpec("Org/a", revision=SHA_A)]
        assert apply_manifest_overrides(specs, None) == specs

    def test_fusion_revision_precedence(self):
        manifest = {"fusion": {"repo_id": "Org/f", "revision": SHA_A}}
        assert resolve_fusion_revision("Org/f", None, None) is None
        assert resolve_fusion_revision("Org/f", None, manifest) == SHA_A
        assert resolve_fusion_revision("Org/f", SHA_B, manifest) == SHA_B
        # Manifest for a different fusion repo is ignored
        assert resolve_fusion_revision("Org/other", None, manifest) is None


class TestManifestHash:
    def test_stable_and_sensitive(self):
        models = [
            {"repo_id": "Org/f", "revision": SHA_A, "weights_file": "f.pkl", "input_size": None},
            {"repo_id": "Org/a", "revision": SHA_B, "weights_file": "a.pth", "input_size": 224},
        ]
        h = compute_manifest_hash(models)
        assert h == compute_manifest_hash([dict(m, name="ignored") for m in models])
        assert h != compute_manifest_hash(list(reversed(models)))
        assert h != compute_manifest_hash([models[0], dict(models[1], input_size=256)])
        assert h != compute_manifest_hash([models[0], dict(models[1], revision=SHA_A)])


# ---------------------------------------------------------------------------
# HF hub cache keyed by revision
# ---------------------------------------------------------------------------

@pytest.fixture
def hf_service(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_home"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "cache"))
    return HFHubService(cache_dir=str(tmp_path / "cache"), token=None)


def _populate(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}")


class TestRevisionKeyedCache:
    def test_local_dir_naming(self, hf_service):
        base = Path(hf_service.cache_dir)
        assert hf_service.local_dir_for("Org/a") == base / "Org--a"
        assert hf_service.local_dir_for("Org/a", SHA_A) == base / f"Org--a@{SHA_A}"
        assert hf_service.local_dir_for("Org/a", "refs/pr/1") == base / "Org--a@refs--pr--1"

    def test_cache_hit_is_per_revision(self, hf_service):
        _populate(hf_service.local_dir_for("Org/a", SHA_A))
        assert hf_service.is_cached("Org/a", SHA_A)
        assert not hf_service.is_cached("Org/a", SHA_B)
        assert not hf_service.is_cached("Org/a")

    def test_new_revision_triggers_download(self, hf_service):
        _populate(hf_service.local_dir_for("Org/a", SHA_A))
        with patch("app.services.hf_hub_service.snapshot_download") as snap:
            snap.side_effect = lambda **kw: kw["local_dir"]
            # Cached revision: no network
            path = hf_service.download_repo("Org/a", SHA_A)
            assert path == str(hf_service.local_dir_for("Org/a", SHA_A))
            snap.assert_not_called()

            # New revision: fresh download into its own directory
            path = hf_service.download_repo("Org/a", SHA_B)
            snap.assert_called_once()
            kwargs = snap.call_args.kwargs
            assert kwargs["revision"] == SHA_B
            assert kwargs["local_dir"] == str(hf_service.local_dir_for("Org/a", SHA_B))
            assert path == kwargs["local_dir"]

    def test_unpinned_uses_legacy_cache(self, hf_service):
        _populate(hf_service.local_dir_for("Org/a"))
        with patch("app.services.hf_hub_service.snapshot_download") as snap:
            assert hf_service.download_repo("Org/a") == str(hf_service.local_dir_for("Org/a"))
            snap.assert_not_called()

    def test_unpinned_download_uses_main(self, hf_service):
        with patch("app.services.hf_hub_service.snapshot_download") as snap:
            snap.side_effect = lambda **kw: kw["local_dir"]
            hf_service.download_repo("Org/a")
            assert snap.call_args.kwargs["revision"] == "main"

    def test_resolve_commit_sha(self, tmp_path):
        assert HFHubService.resolve_commit_sha(str(tmp_path), SHA_A.upper()) == SHA_A
        assert HFHubService.resolve_commit_sha(str(tmp_path), "main") is None

        meta = tmp_path / ".cache" / "huggingface" / "download"
        meta.mkdir(parents=True)
        (meta / "config.json.metadata").write_text(f"{SHA_B}\netag\n123.0\n")
        assert HFHubService.resolve_commit_sha(str(tmp_path), "main") == SHA_B
        assert HFHubService.resolve_commit_sha(str(tmp_path)) == SHA_B


# ---------------------------------------------------------------------------
# Gradfield artifact resolution
# ---------------------------------------------------------------------------

gradfield = pytest.importorskip("app.models.wrappers.gradfield_cnn_wrapper")


class TestGradfieldResolution:
    def test_input_size_from_config(self):
        assert gradfield.resolve_input_size({"input": {"size": 224}}, {"input_size": 256}) == 224

    def test_input_size_from_preprocess(self):
        assert gradfield.resolve_input_size({}, {"input_size": [256, 256]}) == 256
        assert gradfield.resolve_input_size({"input": {"type": "luminance"}}, {"input_size": 192}) == 192

    def test_input_size_default(self):
        assert gradfield.resolve_input_size({}, {}) == 224

    def test_live_repo_config_resolves_to_224(self):
        # Shape of DeepFakeDetector/gradfield-cnn-final config.json (no preprocess.json)
        config = {"input": {"type": "luminance", "channels": 1, "size": 224}}
        assert gradfield.resolve_input_size(config, {}) == 224

    def _weights_dir(self, tmp_path):
        for name in ("gradient_field_cnn_v2.pth", "gradient_field_cnn_v3_finetuned.pth"):
            (tmp_path / name).write_bytes(b"")
        return str(tmp_path)

    def test_weights_legacy_preference(self, tmp_path):
        path = gradfield.resolve_weights_path(self._weights_dir(tmp_path), {})
        assert path.name == "gradient_field_cnn_v3_finetuned.pth"

    def test_weights_from_config(self, tmp_path):
        path = gradfield.resolve_weights_path(
            self._weights_dir(tmp_path), {"weights_file": "gradient_field_cnn_v2.pth"}
        )
        assert path.name == "gradient_field_cnn_v2.pth"

    def test_weights_override_beats_config(self, tmp_path):
        path = gradfield.resolve_weights_path(
            self._weights_dir(tmp_path),
            {"weights_file": "gradient_field_cnn_v3_finetuned.pth"},
            override="gradient_field_cnn_v2.pth",
        )
        assert path.name == "gradient_field_cnn_v2.pth"

    def test_missing_requested_weights_raises(self, tmp_path):
        with pytest.raises(ConfigurationError):
            gradfield.resolve_weights_path(self._weights_dir(tmp_path), {"weights_file": "nope.pth"})

    def test_no_weights_raises(self, tmp_path):
        with pytest.raises(ConfigurationError):
            gradfield.resolve_weights_path(str(tmp_path), {})


# ---------------------------------------------------------------------------
# Registry wiring (fake hub service and wrappers)
# ---------------------------------------------------------------------------

class _FakeHub:
    def __init__(self, root: Path, configs):
        self.root = root
        self.configs = configs
        self.calls = []

    def download_repo(self, repo_id, revision=None, force_download=False):
        self.calls.append((repo_id, revision))
        path = self.root / f"{repo_id.replace('/', '--')}@{revision or 'main'}"
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps(self.configs[repo_id]))
        return str(path)

    resolve_commit_sha = staticmethod(HFHubService.resolve_commit_sha)


def _fake_wrapper_classes():
    from app.models.wrappers.base_wrapper import BaseFusionWrapper, BaseSubmodelWrapper

    class FakeSub(BaseSubmodelWrapper):
        def load(self):
            self.weights_file = self.config.get("weights_file", "default.pth")
            self.input_size = (self.config.get("input") or {}).get("size", 224)
            self._predict_fn = lambda: None

        def predict(self, *a, **k):
            return {}

    class FakeFusion(BaseFusionWrapper):
        def load(self):
            self.weights_file = "fusion.pkl"
            self._predict_fn = lambda: None

        def predict(self, *a, **k):
            return {}

    return FakeSub, FakeFusion


class TestRegistryManifest:
    def test_load_with_object_submodels_and_manifest(self, tmp_path, monkeypatch):
        from app.services import model_registry as mr

        configs = {
            "Org/fusion": {
                "type": "probability_stacking_fusion",
                "submodels": [
                    "Org/a",
                    {"repo_id": "Org/b", "revision": "v1", "weights_file": "b2.pth", "input_size": 256},
                ],
            },
            "Org/a": {"name": "a"},
            "Org/b": {"name": "b", "input": {"size": 224}},
        }
        hub = _FakeHub(tmp_path, configs)
        manifest_path = tmp_path / "model-manifest.json"
        manifest_path.write_text(json.dumps({
            "fusion": {"repo_id": "Org/fusion", "revision": SHA_A},
            "submodels": [{"repo_id": "Org/a", "revision": SHA_B}],
        }))

        FakeSub, FakeFusion = _fake_wrapper_classes()
        monkeypatch.setattr(mr, "get_hf_hub_service", lambda: hub)
        monkeypatch.setattr(mr, "get_wrapper_class", lambda config: FakeSub)
        monkeypatch.setattr(mr, "get_fusion_wrapper_class", lambda config: FakeFusion)
        monkeypatch.setattr(mr.settings, "MODEL_MANIFEST_PATH", str(manifest_path))
        monkeypatch.setattr(mr.settings, "HF_FUSION_REVISION", None)

        registry = mr.ModelRegistry()
        asyncio.run(registry.load_from_fusion_repo("Org/fusion"))

        assert hub.calls == [("Org/fusion", SHA_A), ("Org/a", SHA_B), ("Org/b", "v1")]

        models = {m["repo_id"]: m for m in registry.list_models()}
        assert models["Org/fusion"]["revision"] == SHA_A
        assert models["Org/fusion"]["weights_file"] == "fusion.pkl"
        assert models["Org/a"]["revision"] == SHA_B
        assert models["Org/b"]["requested_revision"] == "v1"
        assert models["Org/b"]["weights_file"] == "b2.pth"
        assert models["Org/b"]["input_size"] == 256

        manifest_hash = registry.get_manifest_hash()
        assert manifest_hash and len(manifest_hash) == 64

        # Explicit fusion revision setting wins over the manifest
        monkeypatch.setattr(mr.settings, "HF_FUSION_REVISION", "pinned")
        hub.calls.clear()
        asyncio.run(registry.load_from_fusion_repo("Org/fusion", force_reload=True))
        assert hub.calls[0] == ("Org/fusion", "pinned")
        assert registry.get_manifest_hash() != manifest_hash


# ---------------------------------------------------------------------------
# Golden-output scaffolding
# ---------------------------------------------------------------------------

class TestGoldenScaffolding:
    def test_synthetic_images_are_deterministic(self, tmp_path):
        from app.scripts.make_golden import list_images, write_synthetic_images

        first = [p.read_bytes() for p in write_synthetic_images(tmp_path / "a", 5)]
        second = [p.read_bytes() for p in write_synthetic_images(tmp_path / "b", 5)]
        assert first == second
        assert [p.name for p in list_images(tmp_path / "a")] == [f"synthetic_{i:02d}.png" for i in range(5)]

    def test_build_expected_with_fake_registry(self, tmp_path):
        from app.scripts.make_golden import build_expected, write_synthetic_images

        class Sub:
            def __init__(self, p):
                self.p = p

            def predict(self, image_bytes=None, explain=False):
                return {"prob_fake": self.p}

        class Fusion:
            def predict(self, outputs):
                return {"prob_fake": sum(o["prob_fake"] for o in outputs.values()) / len(outputs)}

        class Registry:
            def get_all_submodels(self):
                return {"a": Sub(0.2), "b": Sub(0.6)}

            def get_fusion(self):
                return Fusion()

            def get_manifest_hash(self):
                return "h"

            def list_models(self):
                return [{"name": "a", "repo_id": "Org/a", "revision": SHA_A}]

        images = write_synthetic_images(tmp_path, 2)
        expected = build_expected(Registry(), images, tmp_path)
        assert expected["manifest_hash"] == "h"
        assert expected["models"][0]["revision"] == SHA_A
        assert expected["images"]["synthetic_00.png"] == pytest.approx({"a": 0.2, "b": 0.6, "fusion": 0.4})
