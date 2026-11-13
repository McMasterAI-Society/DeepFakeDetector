"""
Unit tests for the foundation-model probe (no downloads: tiny random backbones).

The backbones are the real timm architectures (CLIP ViT and DINOv3 Eva) shrunk
to embed_dim 32, so hooks, RoPE/register tokens and patch-token gradients are
exercised exactly as in production.
"""

import asyncio
import base64
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import app.models.wrappers.foundation_probe_wrapper as fpw
from app.core.config import settings
from app.core.errors import ConfigurationError
from app.models.wrappers.foundation_probe_wrapper import (
    DEFAULT_LOGIT_CLIP,
    FoundationProbeWrapper,
    ProbeHead,
    feature_spec_hash,
    load_head,
    parse_bands,
    verdict_for,
)
from app.models.wrappers.vit_base_wrapper import ViTBaseWrapper
from app.services.model_registry import ModelRegistry, get_wrapper_class

TEMPLATE = Path(__file__).resolve().parents[1] / "hf" / "fm-probe-clip-dinov3"
TINY_DIM = 32
TINY = {
    "clip_b16": dict(name="vit_base_patch16_clip_224", mean=(0.48145466, 0.4578275, 0.40821073),
                     std=(0.26862954, 0.26130258, 0.27577711)),
    "dinov3_b16": dict(name="vit_base_patch16_dinov3", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
}


def tiny_module(key):
    import timm
    torch.manual_seed(0 if key == "clip_b16" else 1)
    return timm.create_model(TINY[key]["name"], pretrained=False, num_classes=0,
                             embed_dim=TINY_DIM, depth=10, num_heads=2)


def tiny_resolver(key, source, search_dirs, download):
    return tiny_module(key), list(TINY[key]["mean"]), list(TINY[key]["std"]), f"tiny:{key}"


def random_head(config, dim, seed=0, **kw):
    rng = np.random.default_rng(seed)
    args = dict(
        scaler_mean=rng.normal(0, 0.05, dim), scaler_scale=rng.uniform(0.02, 0.1, dim),
        coef=rng.normal(0, 0.05, dim), intercept=-0.3, platt_a=0.4, platt_b=-0.6, threshold=0.31,
        head_version="test-head", feature_spec_hash=feature_spec_hash(config),
    )
    args.update(kw)
    return ProbeHead(**args)


@pytest.fixture
def probe_dir(tmp_path):
    shutil.copyfile(TEMPLATE / "config.json", tmp_path / "config.json")
    config = json.loads((tmp_path / "config.json").read_text())
    random_head(config, 2 * 3 * TINY_DIM).save(tmp_path / "head.npz")
    return tmp_path


@pytest.fixture
def tiny_backbones(monkeypatch):
    monkeypatch.setattr(FoundationProbeWrapper, "backbone_resolver", staticmethod(tiny_resolver))
    monkeypatch.setattr(settings, "PROBE_DEVICE", "cpu")
    monkeypatch.setattr(settings, "PROBE_VERDICT_BANDS", None)


@pytest.fixture
def wrapper(probe_dir, tiny_backbones):
    w = FoundationProbeWrapper("test/probe", json.loads((probe_dir / "config.json").read_text()), str(probe_dir))
    w.load()
    return w


def jpeg_bytes(w=320, h=240, seed=0):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h // 8, w // 8, 3), dtype=np.uint8)
    im = Image.fromarray(arr).resize((w, h), Image.BICUBIC)
    b = io.BytesIO()
    im.save(b, format="JPEG", quality=95)
    return b.getvalue()


# --------------------------------------------------------------------------- #
# Registry dispatch
# --------------------------------------------------------------------------- #

class TestDispatch:
    def test_probe_type_wins_over_vit_name(self):
        cfg = {"type": "foundation-probe", "arch": "vit_base_patch16_clip_224",
               "model_name": "vit_base_patch16_dinov3"}
        assert get_wrapper_class(cfg) is FoundationProbeWrapper

    def test_template_config_dispatches_to_probe(self):
        assert get_wrapper_class(json.loads((TEMPLATE / "config.json").read_text())) is FoundationProbeWrapper

    def test_plain_vit_still_vit(self):
        assert get_wrapper_class({"arch": "vit_base_patch16_224"}) is ViTBaseWrapper


# --------------------------------------------------------------------------- #
# Head format
# --------------------------------------------------------------------------- #

class TestHeadFormat:
    def config(self):
        return json.loads((TEMPLATE / "config.json").read_text())

    def test_roundtrip(self, tmp_path):
        h = random_head(self.config(), 12)
        h.save(tmp_path / "h.npz")
        g = load_head(tmp_path / "h.npz", expected_dim=12, expected_spec_hash=feature_spec_hash(self.config()))
        assert np.array_equal(g.coef, h.coef) and g.threshold == h.threshold
        assert g.head_version == "test-head" and g.logit_clip == pytest.approx(DEFAULT_LOGIT_CLIP)
        x = np.random.default_rng(1).normal(size=(5, 12))
        assert np.allclose(g.calibrate(g.raw_logit(x)), h.calibrate(h.raw_logit(x)))

    def test_npz_has_no_object_arrays(self, tmp_path):
        random_head(self.config(), 4).save(tmp_path / "h.npz")
        with np.load(tmp_path / "h.npz", allow_pickle=False) as d:
            assert all(d[k].dtype != object for k in d.files)

    def test_pickled_array_rejected(self, tmp_path):
        d = random_head(self.config(), 4).to_npz_dict()
        d["coef"] = np.array([{"evil": 1}], dtype=object)
        np.savez(tmp_path / "h.npz", **d)
        with pytest.raises(ConfigurationError):
            load_head(tmp_path / "h.npz")

    def test_missing_key(self, tmp_path):
        d = random_head(self.config(), 4).to_npz_dict()
        del d["platt_a"]
        np.savez(tmp_path / "h.npz", **d)
        with pytest.raises(ConfigurationError, match="missing"):
            load_head(tmp_path / "h.npz")

    def test_dim_mismatch(self, tmp_path):
        random_head(self.config(), 4).save(tmp_path / "h.npz")
        with pytest.raises(ConfigurationError, match="features"):
            load_head(tmp_path / "h.npz", expected_dim=8)

    def test_spec_hash_mismatch(self, tmp_path):
        random_head(self.config(), 4, feature_spec_hash="0" * 64).save(tmp_path / "h.npz")
        with pytest.raises(ConfigurationError, match="feature spec"):
            load_head(tmp_path / "h.npz", expected_spec_hash=feature_spec_hash(self.config()))

    def test_spec_hash_tracks_layers(self):
        cfg = self.config()
        other = dict(cfg, layers=[4, 8])
        assert feature_spec_hash(cfg) != feature_spec_hash(other)

    @pytest.mark.parametrize("field,value", [("coef", np.array([np.nan, 0, 0, 0])),
                                             ("scaler_scale", np.zeros(4)), ("threshold", 1.5)])
    def test_invalid_values(self, tmp_path, field, value):
        d = random_head(self.config(), 4).to_npz_dict()
        d[field] = value
        np.savez(tmp_path / "h.npz", **d)
        with pytest.raises(ConfigurationError):
            load_head(tmp_path / "h.npz")

    def test_legacy_head_without_metadata(self, tmp_path):
        d = random_head(self.config(), 4).to_npz_dict()
        for k in ("format_version", "head_version", "feature_spec_hash", "logit_clip"):
            del d[k]
        np.savez(tmp_path / "h.npz", **d)
        h = load_head(tmp_path / "h.npz", expected_spec_hash="anything")
        assert h.head_version == "unversioned" and h.logit_clip == pytest.approx(DEFAULT_LOGIT_CLIP)

    def test_logit_clip_bounds_probability(self):
        h = random_head(self.config(), 4)
        lo, hi = h.calibrate(np.array([-1e3, 1e3]))
        assert lo == pytest.approx(1 / (1 + np.exp(-(h.platt_a * -DEFAULT_LOGIT_CLIP + h.platt_b))))
        assert hi == pytest.approx(1 / (1 + np.exp(-(h.platt_a * DEFAULT_LOGIT_CLIP + h.platt_b))))


# --------------------------------------------------------------------------- #
# Verdict bands
# --------------------------------------------------------------------------- #

class TestVerdict:
    @pytest.mark.parametrize("p,expected", [(0.0, "real"), (0.199, "real"), (0.2, "inconclusive"),
                                            (0.5, "inconclusive"), (0.699, "inconclusive"),
                                            (0.7, "ai"), (1.0, "ai")])
    def test_edges(self, p, expected):
        assert verdict_for(p, (0.2, 0.7)) == expected

    def test_empty_band_is_binary(self):
        assert verdict_for(0.5, (0.5, 0.5)) == "ai" and verdict_for(0.49, (0.5, 0.5)) == "real"

    @pytest.mark.parametrize("value", ["0.7,0.2", "-0.1,0.5", "0.2,1.5"])
    def test_invalid(self, value):
        with pytest.raises(ConfigurationError):
            parse_bands(value)

    def test_parse(self):
        assert parse_bands("0.15, 0.6") == (0.15, 0.6)
        assert parse_bands([0.2, 0.7]) == (0.2, 0.7)
        assert parse_bands(None) is None

    def test_setting_overrides_config(self, probe_dir, tiny_backbones, monkeypatch):
        monkeypatch.setattr(settings, "PROBE_VERDICT_BANDS", "0.1,0.9")
        w = FoundationProbeWrapper("t/p", json.loads((probe_dir / "config.json").read_text()), str(probe_dir))
        w.load()
        assert w.bands == (0.1, 0.9)
        out = w.predict(image_bytes=jpeg_bytes())
        assert out["verdict"] == verdict_for(out["prob_fake"], (0.1, 0.9))


# --------------------------------------------------------------------------- #
# Wrapper with tiny backbones
# --------------------------------------------------------------------------- #

class TestWrapper:
    def test_load(self, wrapper):
        assert wrapper.is_loaded()
        assert [b.key for b in wrapper.backbones] == ["clip_b16", "dinov3_b16"]
        assert wrapper.backbones[1].npt == 5  # DINOv3: CLS + 4 registers
        assert wrapper.threshold == pytest.approx(0.31)  # from head.npz
        assert wrapper.model_version.startswith("fm-probe-clip-dinov3:test-head:")

    def test_threshold_comes_from_head(self, probe_dir, tiny_backbones):
        cfg = json.loads((probe_dir / "config.json").read_text())
        cfg["threshold"] = 0.9
        w = FoundationProbeWrapper("t/p", cfg, str(probe_dir))
        w.load()
        assert w.threshold == pytest.approx(0.31)

    def test_predict_contract(self, wrapper):
        out = wrapper.predict(image_bytes=jpeg_bytes())
        assert 0.0 <= out["prob_fake"] <= 1.0
        assert out["pred_int"] == int(out["prob_fake"] >= wrapper.threshold)
        assert out["pred"] == ("fake" if out["pred_int"] else "real")
        assert out["verdict"] == verdict_for(out["prob_fake"], wrapper.bands)
        assert set(out["backbones"]) == {"clip_b16", "dinov3_b16"}
        assert "heatmap_base64" not in out

    def test_contributions_sum_to_logit(self, wrapper):
        out = wrapper.predict(image_bytes=jpeg_bytes(seed=3))
        bbs = out["backbones"]
        total = sum(b["logit_contribution"] for b in bbs.values()) + out["meta"]["intercept"]
        assert total == pytest.approx(out["meta"]["raw_logit"], abs=1e-9)
        assert sum(b["contribution_percentage"] for b in bbs.values()) == pytest.approx(100.0)

    def test_features_match_manual_computation(self, wrapper):
        from app.models.wrappers.foundation_probe_wrapper import preprocess
        im = Image.open(io.BytesIO(jpeg_bytes(seed=5)))
        x = torch.from_numpy(preprocess(im)).permute(2, 0, 1)[None].float() / 255
        parts = []
        for bb in wrapper.backbones:
            cls = {}
            hooks = [bb.m.blocks[i].register_forward_hook(
                lambda m, i_, o, i=i: cls.__setitem__(i, o[:, 0])) for i in (5, 8)]
            with torch.no_grad():
                xn = (x - bb.mean) / bb.std
                pooled = bb.m.forward_head(bb.m.forward_features(xn), pre_logits=True)
            for h in hooks:
                h.remove()
            for t in (cls[5], cls[8], pooled):
                parts.append(torch.nn.functional.normalize(t, dim=-1)[0].numpy())
        f = np.concatenate(parts)
        h = wrapper.head
        expected = float(h.calibrate(h.raw_logit(f)))
        assert wrapper.predict(image=im)["prob_fake"] == pytest.approx(expected, abs=1e-6)

    def test_explain_heatmaps(self, wrapper):
        out = wrapper.predict(image_bytes=jpeg_bytes(400, 200), explain=True)
        for entry in [out] + list(out["backbones"].values()):
            assert entry["explainability_type"] == "input_gradient"
            png = Image.open(io.BytesIO(base64.b64decode(entry["heatmap_base64"])))
            assert png.size == (448, 224)  # original 2:1 aspect, short side 224
            assert isinstance(entry["focus_summary"], str)
        plain = wrapper.predict(image_bytes=jpeg_bytes(400, 200))
        assert out["prob_fake"] == pytest.approx(plain["prob_fake"], abs=1e-6)

    def test_relevance_is_finite_and_nonzero(self, wrapper):
        from app.models.wrappers.foundation_probe_wrapper import preprocess
        x = wrapper._to_tensor(preprocess(Image.open(io.BytesIO(jpeg_bytes()))))
        rel = wrapper._forward(x, explain=True)["relevance"]
        for r in rel.values():
            assert r.shape == (14, 14) and np.isfinite(r).all() and np.abs(r).sum() > 0

    def test_exif_orientation(self, wrapper):
        im = Image.open(io.BytesIO(jpeg_bytes(400, 200)))
        exif = im.getexif()
        exif[0x0112] = 6  # rotate 90: displayed as 200x400
        b = io.BytesIO()
        im.save(b, format="JPEG", exif=exif)
        out = wrapper.predict(image_bytes=b.getvalue(), explain=True)
        png = Image.open(io.BytesIO(base64.b64decode(out["heatmap_base64"])))
        assert png.size == (224, 448)

    def test_rgba_and_grayscale_inputs(self, wrapper):
        for mode in ("RGBA", "L", "P"):
            im = Image.new(mode, (300, 260))
            assert 0 <= wrapper.predict(image=im)["prob_fake"] <= 1

    def test_head_dim_mismatch_fails_load(self, probe_dir, tiny_backbones):
        cfg = json.loads((probe_dir / "config.json").read_text())
        random_head(cfg, 10).save(probe_dir / "head.npz")
        w = FoundationProbeWrapper("t/p", cfg, str(probe_dir))
        with pytest.raises(ConfigurationError):
            w.load()


# --------------------------------------------------------------------------- #
# Backbone weight resolution (no network)
# --------------------------------------------------------------------------- #

class TestBackboneResolution:
    def test_local_pt_is_loaded_without_pickle(self, tmp_path, monkeypatch):
        m = tiny_module("clip_b16")
        sd = {k: v.half() for k, v in m.state_dict().items()}
        torch.save({"name": "tiny", "state_dict": sd, "cfg": {"mean": (0.1, 0.2, 0.3), "std": (1.0, 1.0, 1.0)}},
                   tmp_path / "clip_b16.fp16.pt")
        monkeypatch.setattr(fpw, "_timm_create", lambda name, file=None: tiny_module("clip_b16"))
        mod, mean, std, where = fpw.resolve_backbone_module(
            "clip_b16", {"timm_name": "tiny"}, [tmp_path], download=lambda *a: pytest.fail("downloaded"))
        assert where.startswith("local:") and mean == [0.1, 0.2, 0.3]
        assert torch.equal(mod.state_dict()["cls_token"], sd["cls_token"].float())

    def test_upstream_uses_pinned_revision(self, tmp_path, monkeypatch):
        calls, created = [], []
        monkeypatch.setattr(fpw, "_timm_create", lambda name, file=None: created.append((name, file)) or tiny_module("clip_b16"))
        src = dict(fpw.DEFAULT_BACKBONE_SOURCES["dinov3_b16"])
        fpw.resolve_backbone_module("dinov3_b16", src, [tmp_path],
                                    download=lambda r, f, rev: calls.append((r, f, rev)) or "/x/model.safetensors")
        assert calls == [("timm/vit_base_patch16_dinov3.lvd1689m", "model.safetensors",
                          "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da")]
        assert created == [("vit_base_patch16_dinov3.lvd1689m", "/x/model.safetensors")]


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

class TestRegistry:
    def load(self, probe_dir, manifest=None, tmp_path=None, monkeypatch=None):
        if manifest is not None:
            path = tmp_path / "manifest.json"
            path.write_text(json.dumps(manifest))
            monkeypatch.setattr(settings, "MODEL_MANIFEST_PATH", str(path))
        reg = ModelRegistry()
        asyncio.run(reg.load_foundation_probe(local_path=str(probe_dir)))
        return reg

    def test_local_load(self, probe_dir, tiny_backbones, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "MODEL_MANIFEST_PATH", None)
        reg = self.load(probe_dir, manifest={}, tmp_path=tmp_path, monkeypatch=monkeypatch)
        assert reg.mode == "foundation-probe"
        assert reg.get_submodel_names() == ["fm-probe-clip-dinov3"]
        assert reg.get_submodel("foundation-probe") is reg.get_probe()
        models = reg.list_models()
        assert [m["model_type"] for m in models] == ["probe", "backbone", "backbone"]
        assert models[0]["requested_revision"].startswith("local:")
        assert models[2]["revision"] == "c6a5fb7d12bbd3cf3b0079253141c3332aaed7da"
        assert len(reg.get_manifest_hash()) == 64

    def test_manifest_pins_backbones(self, probe_dir, tiny_backbones, monkeypatch, tmp_path):
        manifest = {"probe": {"repo_id": settings.HF_PROBE_REPO_ID, "revision": None,
                              "backbones": {"clip_b16": {"revision": "a" * 40}}}}
        reg = self.load(probe_dir, manifest, tmp_path, monkeypatch)
        clip = [m for m in reg.list_models() if m["name"] == "clip-b16"][0]
        assert clip["revision"] == "a" * 40
        assert clip["repo_id"] == "timm/vit_base_patch16_clip_224.openai"

    def test_hash_changes_with_backbone_pin(self, probe_dir, tiny_backbones, monkeypatch, tmp_path):
        h1 = self.load(probe_dir, {}, tmp_path, monkeypatch).get_manifest_hash()
        h2 = self.load(probe_dir, {"probe": {"backbones": {"clip_b16": {"revision": "b" * 40}}}},
                       tmp_path, monkeypatch).get_manifest_hash()
        assert h1 != h2

    def test_repo_manifest_file_is_valid(self):
        from app.services.model_manifest import load_local_manifest, resolve_probe_pins
        path = Path(__file__).resolve().parents[1] / "model-manifest.json"
        pins = resolve_probe_pins(settings.HF_PROBE_REPO_ID, load_local_manifest(str(path)))
        assert pins["revision"] is None  # placeholder until the head repo is uploaded
        assert pins["backbones"]["dinov3_b16"]["revision"] == fpw.DEFAULT_BACKBONE_SOURCES["dinov3_b16"]["revision"]
        assert pins["backbones"]["clip_b16"]["revision"] == fpw.DEFAULT_BACKBONE_SOURCES["clip_b16"]["revision"]

    def test_non_probe_config_rejected(self, tmp_path, tiny_backbones, monkeypatch):
        (tmp_path / "config.json").write_text(json.dumps({"arch": "vit_base_patch16_224"}))
        monkeypatch.setattr(settings, "MODEL_MANIFEST_PATH", None)
        with pytest.raises(ConfigurationError):
            asyncio.run(ModelRegistry().load_foundation_probe(local_path=str(tmp_path)))

    def test_fallback_to_legacy(self, monkeypatch):
        reg = ModelRegistry()
        called = []

        async def boom(*a, **k):
            raise RuntimeError("head repo missing")

        async def legacy(repo_id, **k):
            called.append(repo_id)

        monkeypatch.setattr(reg, "load_foundation_probe", boom)
        monkeypatch.setattr(reg, "load_from_fusion_repo", legacy)
        monkeypatch.setattr(settings, "MODEL_MODE", "foundation-probe")
        monkeypatch.setattr(settings, "MODEL_FALLBACK_TO_LEGACY", True)
        asyncio.run(reg.load_production_model())
        assert called == [settings.HF_FUSION_REPO_ID]
        monkeypatch.setattr(settings, "MODEL_FALLBACK_TO_LEGACY", False)
        with pytest.raises(RuntimeError):
            asyncio.run(reg.load_production_model())

    def test_legacy_mode_skips_probe(self, monkeypatch):
        reg = ModelRegistry()
        called = []

        async def legacy(repo_id, **k):
            called.append(repo_id)

        monkeypatch.setattr(reg, "load_foundation_probe", lambda *a, **k: pytest.fail("probe loaded"))
        monkeypatch.setattr(reg, "load_from_fusion_repo", legacy)
        monkeypatch.setattr(settings, "MODEL_MODE", "legacy-fusion")
        asyncio.run(reg.load_production_model())
        assert called == [settings.HF_FUSION_REPO_ID]


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@pytest.fixture
def probe_client(probe_dir, tiny_backbones, monkeypatch):
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    import app.services.model_registry as mr

    monkeypatch.setattr(settings, "MODEL_MANIFEST_PATH", None)
    reg = ModelRegistry()
    asyncio.run(reg.load_foundation_probe(local_path=str(probe_dir)))
    monkeypatch.setattr(mr, "_model_registry", reg)
    for mod in ("app.services.inference_service", "app.services.fusion_service"):
        monkeypatch.setattr(f"{mod}._{mod.rsplit('.', 1)[1]}", None)
    with patch("app.services.model_registry.ModelRegistry.load_from_fusion_repo", new_callable=AsyncMock):
        from app.main import app
    return TestClient(app), reg


class TestApi:
    def post(self, client, path, **params):
        files = {"image": ("x.jpg", jpeg_bytes(), "image/jpeg")}
        return client.post(path, params=params, files=files)

    def test_predict_probe_response(self, probe_client, monkeypatch):
        client, reg = probe_client
        monkeypatch.setattr(settings, "PROBE_EXPLAIN_ON_PREDICT", False)
        r = self.post(client, "/predict", use_fusion="true", return_submodels="true", explain="true")
        assert r.status_code == 200, r.text
        d = r.json()
        assert d["verdict"] in ("ai", "inconclusive", "real")
        assert d["model_version"] == reg.get_probe().model_version
        assert d["verdict_bands"] == [0.2, 0.7]
        assert set(d["submodels"]) == {"clip-b16", "dinov3-b16"}
        assert set(d["model_display_info"]) == {"clip-b16", "dinov3-b16"}
        meta = d["fusion_meta"]
        assert sum(meta["weighted_contributions"].values()) + meta["logit_intercept"] == pytest.approx(
            reg.get_probe().predict(image_bytes=jpeg_bytes())["meta"]["raw_logit"], abs=1e-6)
        assert d["final"]["heatmap_base64"] is None  # explain is off the hot path by default
        assert d["timing_ms"]["inference"] is not None

    def test_predict_with_explain_enabled(self, probe_client, monkeypatch):
        client, _ = probe_client
        monkeypatch.setattr(settings, "PROBE_EXPLAIN_ON_PREDICT", True)
        d = self.post(client, "/predict", explain="true").json()
        assert d["final"]["explainability_type"] == "input_gradient"
        assert all(s["heatmap_base64"] for s in d["submodels"].values())

    def test_explain_endpoint(self, probe_client):
        client, _ = probe_client
        d = self.post(client, "/explain").json()
        assert d["final"]["heatmap_base64"] and d["verdict"]
        assert all(s["explainability_type"] == "input_gradient" for s in d["submodels"].values())

    def test_single_model_path(self, probe_client):
        client, _ = probe_client
        assert self.post(client, "/predict", use_fusion="false").status_code == 200
        assert self.post(client, "/predict", use_fusion="false", model="vit-base").status_code == 404

    def test_models_and_health(self, probe_client):
        client, _ = probe_client
        m = client.get("/models").json()
        assert m["fusion"]["model_type"] == "probe"
        assert [s["name"] for s in m["submodels"]] == ["clip-b16", "dinov3-b16"]
        h = client.get("/health").json()
        assert h["manifest_hash"] and len(h["models"]) == 3
        ready = client.get("/ready").json()
        assert ready["status"] == "ready" and len(ready["submodels"]) == 2

    def test_explain_endpoint_needs_probe_mode(self, monkeypatch):
        from unittest.mock import AsyncMock, patch
        from fastapi.testclient import TestClient
        import app.services.model_registry as mr
        monkeypatch.setattr(mr, "_model_registry", ModelRegistry())
        with patch("app.services.model_registry.ModelRegistry.load_from_fusion_repo", new_callable=AsyncMock):
            from app.main import app
        r = TestClient(app).post("/explain", files={"image": ("x.jpg", jpeg_bytes(), "image/jpeg")})
        assert r.status_code == 409


# --------------------------------------------------------------------------- #
# Export script
# --------------------------------------------------------------------------- #

class TestExport:
    def make_joblib(self, tmp_path, dim):
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, dim))
        y = (X[:, 0] + 0.5 * rng.normal(size=200) > 0).astype(int)
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.1)).fit(X, y)
        path = tmp_path / "head.joblib"
        joblib.dump({"head": pipe, "calib": [0.5, -0.2], "threshold": 0.4,
                     "config": {"bbs": "clip_b16+dinov3_b16", "layers": "multi"}}, path)
        return path, pipe, X

    def test_export_roundtrip(self, tmp_path):
        from app.scripts.export_probe_head import main
        path, pipe, X = self.make_joblib(tmp_path, 2 * 3 * TINY_DIM)
        out = tmp_path / "out"
        assert main(["--joblib", str(path), "--out", str(out), "--head-version", "t1",
                     "--verdict-bands", "0.1,0.9"]) == 0
        cfg = json.loads((out / "config.json").read_text())
        head = load_head(out / "head.npz", expected_spec_hash=feature_spec_hash(cfg))
        assert head.threshold == 0.4 and cfg["threshold"] == 0.4 and head.head_version == "t1"
        assert cfg["verdict_bands"] == [0.1, 0.9]
        assert np.allclose(head.raw_logit(X), pipe.decision_function(X))
        assert (out / "LICENSE-DINOv3.md").read_text(encoding="utf-8").startswith("# DINOv3 License")
        assert "dinov3-license" in (out / "README.md").read_text(encoding="utf-8")

    def test_bands_from_val_preds(self, tmp_path):
        from app.scripts.export_probe_head import bands_from_val
        rng = np.random.default_rng(0)
        rows = [(0, q) for q in rng.uniform(0, 0.5, 1000)] + [(1, q) for q in rng.uniform(0.3, 1, 1000)]
        val = tmp_path / "val.csv"
        val.write_text("label,prob_fake\n" + "\n".join(f"{l},{q}" for l, q in rows))
        lo, hi = bands_from_val(val)
        assert lo == pytest.approx(0.32, abs=0.01) and hi == pytest.approx(0.475, abs=0.01)

    def test_export_rejects_threshold_outside_bands(self, tmp_path):
        from app.scripts.export_probe_head import main
        path, _, _ = self.make_joblib(tmp_path, 4)
        with pytest.raises(SystemExit):
            main(["--joblib", str(path), "--out", str(tmp_path / "o"), "--head-version", "t",
                  "--verdict-bands", "0.5,0.9"])

    def test_export_rejects_wrong_backbones(self, tmp_path):
        import joblib
        from app.scripts.export_probe_head import main
        path, pipe, _ = self.make_joblib(tmp_path, 4)
        d = joblib.load(path)
        d["config"]["bbs"] = "dinov3_b16"
        joblib.dump(d, path)
        with pytest.raises(SystemExit):
            main(["--joblib", str(path), "--out", str(tmp_path / "o"), "--head-version", "t"])
