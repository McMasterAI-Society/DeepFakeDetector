"""
Explainability method tests with small randomly initialised models.

No weights are downloaded: every model is built locally with random init.
"""

import base64
import io

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from app.services import explainability as ex

timm = pytest.importorskip("timm")
from timm.models.vision_transformer import VisionTransformer  # noqa: E402
from timm.models.deit import VisionTransformerDistilled  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


class TinyCNN(nn.Module):
    """conv -> relu -> conv -> relu -> GAP -> linear(2)."""

    def __init__(self, out=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 3, padding=1), nn.ReLU(),
            nn.Conv2d(6, 8, 3, padding=1), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, out)

    def forward(self, x):
        return self.fc(self.pool(self.features(x)).flatten(1))


def tiny_vit(**kw):
    return VisionTransformer(
        img_size=32, patch_size=8, embed_dim=32, depth=3, num_heads=2,
        num_classes=2, **kw,
    ).eval()


def tiny_deit():
    return VisionTransformerDistilled(
        img_size=32, patch_size=8, embed_dim=32, depth=3, num_heads=2, num_classes=2,
    ).eval()


# ---------------------------------------------------------------------------
# CNN CAM
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ex.CAM_METHODS)
def test_cam_shapes_and_ranges(method):
    model = TinyCNN().eval()
    x = torch.randn(1, 3, 16, 16)
    r = ex.compute_cam(model, x, model.features[3], method=method, output_size=(20, 24))
    assert r.heatmap.shape == (20, 24) and r.signed_map.shape == (20, 24)
    assert r.cam.shape == (16, 16)
    assert r.heatmap.min() >= 0 and r.heatmap.max() <= 1
    assert r.signed_map.min() >= -1 and r.signed_map.max() <= 1
    assert r.target_class in (0, 1)
    # Logits come from the same pass and match a plain forward
    with torch.no_grad():
        torch.testing.assert_close(r.logits, model(x))


def test_cam_is_class_specific_and_antisymmetric():
    model = TinyCNN().eval()
    x = torch.randn(1, 3, 16, 16)
    layer = model.features[3]
    r0 = ex.compute_cam(model, x, layer, method="hirescam", target_class=0)
    r1 = ex.compute_cam(model, x, layer, method="hirescam", target_class=1)
    # The 2-class log-odds score flips sign between classes, so does the map
    np.testing.assert_allclose(r0.cam, -r1.cam, rtol=1e-5, atol=1e-6)
    assert not np.allclose(r0.heatmap, r1.heatmap)


def test_hirescam_decomposes_logit_exactly():
    """Last ReLU -> GAP -> linear: the HiResCAM map sums to the score minus bias."""
    model = TinyCNN().eval()
    x = torch.randn(1, 3, 16, 16)
    r = ex.compute_cam(model, x, model.features[3], method="hirescam", target_class=1)
    with torch.no_grad():
        logits = model(x)[0]
    score = (logits[1] - logits[0]).item()
    bias = (model.fc.bias[1] - model.fc.bias[0]).item()
    assert r.cam.sum() == pytest.approx(score - bias, rel=1e-4, abs=1e-5)


def test_cam_works_under_no_grad_and_frozen_params():
    model = TinyCNN().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        r = ex.compute_cam(model, x, model.features[1])
    assert np.isfinite(r.heatmap).all()
    assert all(p.grad is None for p in model.parameters())


def test_cam_does_not_leave_parameter_grads_or_hooks():
    model = TinyCNN().eval()
    layer = model.features[3]
    ex.compute_cam(model, torch.randn(1, 3, 16, 16), layer)
    assert all(p.grad is None for p in model.parameters())
    assert len(layer._forward_hooks) == 0


def test_legacy_gradcam_class_still_works():
    model = TinyCNN().eval()
    hm = ex.GradCAM(model, model.features[3])(torch.randn(1, 3, 16, 16), target_class=1, output_size=(8, 8))
    assert hm.shape == (8, 8) and 0 <= hm.min() <= hm.max() <= 1


def test_class_score_single_logit():
    lg = torch.tensor([[1.5]])
    assert ex.class_score(lg, 1).item() == pytest.approx(1.5)
    assert ex.class_score(lg, 0).item() == pytest.approx(-1.5)
    assert ex.predicted_class(lg) == 1
    assert ex.predicted_class(torch.tensor([[0.0, 1.0]]), threshold=0.9) == 0


# ---------------------------------------------------------------------------
# Transformer attribution
# ---------------------------------------------------------------------------

def test_capture_attention_probs_matches_model_and_restores_flag():
    model = tiny_vit(qk_norm=True)  # q_norm path that manual qkv code used to miss
    x = torch.randn(1, 3, 32, 32)
    attn_mods = [b.attn for b in model.blocks]
    flags = [m.fused_attn for m in attn_mods]
    with torch.no_grad():
        ref = model(x)
        with ex.capture_attention_probs(attn_mods) as attns:
            out = model(x)
    assert [m.fused_attn for m in attn_mods] == flags
    assert len(attns) == len(model.blocks)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)
    for a in attns:
        assert a.shape == (1, 2, 17, 17)
        torch.testing.assert_close(a.sum(-1), torch.ones(1, 2, 17))


@pytest.mark.parametrize("factory,prefix,readout", [(tiny_vit, 1, [0]), (tiny_deit, 2, [0, 1])])
def test_transformer_attribution_shapes(factory, prefix, readout):
    model = factory()
    assert model.num_prefix_tokens == prefix
    x = torch.randn(1, 3, 32, 32)
    r = ex.compute_transformer_attribution(
        model, x, model.blocks, num_prefix_tokens=prefix, readout_tokens=readout,
        grid_size=(4, 4), output_size=(32, 32),
    )
    assert r.method == "grad_rollout"
    assert r.grid.shape == (4, 4) and r.heatmap.shape == (32, 32)
    assert 0 <= r.heatmap.min() <= r.heatmap.max() <= 1
    with torch.no_grad():
        torch.testing.assert_close(r.logits, model(x), rtol=1e-4, atol=1e-5)
    assert all(p.grad is None for p in model.parameters())
    assert all(len(b.attn.attn_drop._forward_hooks) == 0 for b in model.blocks)


def test_grad_rollout_is_class_specific_but_rollout_is_not():
    model = tiny_vit()
    x = torch.randn(1, 3, 32, 32)
    kw = dict(num_prefix_tokens=1, grid_size=(4, 4))
    g0 = ex.compute_transformer_attribution(model, x, model.blocks, target_class=0, **kw).grid
    g1 = ex.compute_transformer_attribution(model, x, model.blocks, target_class=1, **kw).grid
    r0 = ex.compute_transformer_attribution(model, x, model.blocks, method="rollout", target_class=0, **kw).grid
    r1 = ex.compute_transformer_attribution(model, x, model.blocks, method="rollout", target_class=1, **kw).grid
    assert not np.allclose(g0, g1)
    np.testing.assert_allclose(r0, r1)


def test_gradient_weighted_rollout_identity_case():
    """Zero gradients leave R = I, so the cls row has no patch relevance."""
    A = [torch.softmax(torch.randn(1, 2, 5, 5), -1) for _ in range(2)]
    G = [torch.zeros_like(a) for a in A]
    grid = ex.gradient_weighted_rollout(A, G, num_prefix_tokens=1, grid_size=(2, 2))
    assert grid.shape == (2, 2) and np.all(grid == 0)


def test_transformer_falls_back_to_qkv_rollout_when_capture_fails(monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def capture_nothing(_modules):
        yield []  # simulate an attention module whose probabilities can't be hooked

    monkeypatch.setattr(ex, "capture_attention_probs", capture_nothing)
    model = tiny_vit()
    r = ex.compute_transformer_attribution(model, torch.randn(1, 3, 32, 32), model.blocks, grid_size=(4, 4))
    assert r.method == "rollout_qkv"
    assert r.grid.shape == (4, 4)


def test_legacy_attention_rollout_api():
    attn = torch.softmax(torch.randn(3, 197, 197), dim=-1)
    out = ex.attention_rollout(attn, num_prefix_tokens=1)
    assert out.shape == (14, 14)
    assert ex.attention_rollout([], num_prefix_tokens=1).shape == (14, 14)


# ---------------------------------------------------------------------------
# Wrappers (random init, load() bypassed)
# ---------------------------------------------------------------------------

def _image(size=64):
    return Image.fromarray(np.random.randint(0, 255, (size, size, 3), dtype=np.uint8))


def _decode_png(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _prep(wrapper, model, transform=None):
    wrapper._device = CPU
    wrapper._model = model.to(CPU).eval()
    wrapper._transform = transform
    wrapper._predict_fn = wrapper._run_inference
    return wrapper


IMAGENET_TF = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def test_cnn_transfer_wrapper_single_pass():
    from torchvision.models import efficientnet_b0
    from app.models.wrappers.cnn_transfer_wrapper import CNNTransferWrapper

    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.2, inplace=True), nn.Linear(1280, 2))
    w = _prep(CNNTransferWrapper("x/cnn-transfer", {}, "."), m, IMAGENET_TF)
    img = _image()
    plain = w.predict(image=img)
    out = w.predict(image=img, explain=True)
    assert out["prob_fake"] == pytest.approx(plain["prob_fake"], abs=1e-5)
    assert out["explainability_type"] == "grad_cam"
    assert out["explainability_method"] == "hirescam"
    assert _decode_png(out["heatmap_base64"]).size == (224, 224)
    assert "likely face" not in out["focus_summary"]
    # The default CAM layer is the 14x14 stage
    assert w._cam_settings()[0] is m.features[5]


def test_vit_wrapper_grad_rollout():
    from app.models.wrappers.vit_base_wrapper import ViTBaseWrapper, ViTWithMLPHead

    m = ViTWithMLPHead(arch="vit_tiny_patch16_224", hidden_dim=16)
    w = _prep(ViTBaseWrapper("x/vit-base", {}, "."), m, IMAGENET_TF)
    img = _image()
    plain = w.predict(image=img)
    out = w.predict(image=img, explain=True)
    assert out["prob_fake"] == pytest.approx(plain["prob_fake"], abs=1e-4)
    assert out["explainability_type"] == "attention_rollout"
    assert out["explainability_method"] == "grad_rollout"
    assert w._attribution_settings()["readout_tokens"] == [0]
    assert _decode_png(out["heatmap_base64"]).size == (224, 224)


def test_deit_wrapper_uses_cls_and_dist_tokens():
    from app.models.wrappers.deit_distilled_wrapper import DeiTDistilledWrapper, create_custom_mlp_head

    m = timm.create_model("deit_tiny_distilled_patch16_224", pretrained=False, num_classes=0)
    m.head = create_custom_mlp_head(192, 2)
    m.head_dist = create_custom_mlp_head(192, 2)
    w = _prep(DeiTDistilledWrapper("x/deit-distilled", {"explain_method": "rollout"}, "."), m, IMAGENET_TF)
    s = w._attribution_settings()
    assert s["num_prefix_tokens"] == 2 and s["readout_tokens"] == [0, 1]
    assert s["method"] == "rollout"
    out = w.predict(image=_image(), explain=True)
    assert out["explainability_method"] == "rollout"
    assert _decode_png(out["heatmap_base64"]).size == (224, 224)


def test_gradfield_wrapper_signed_map():
    from app.models.wrappers.gradfield_cnn_wrapper import CompactGradientNet, GradfieldCNNWrapper

    model = CompactGradientNet(depth=2, base_filters=4, dropout=0.3, embedding_dim=8)
    w = _prep(GradfieldCNNWrapper("x/gradfield-cnn", {}, "."), model)
    w._resize = transforms.Resize((64, 64))
    img = _image(80)
    plain = w.predict(image=img)
    out = w.predict(image=img, explain=True)
    assert out["prob_fake"] == pytest.approx(plain["prob_fake"], abs=1e-5)
    assert out["explainability_type"] == "signed_grad_cam"
    assert out["explainability_method"] == "hirescam_signed"
    assert _decode_png(out["heatmap_base64"]).size == (64, 64)
    assert _decode_png(out["heatmap_signed_base64"]).size == (64, 64)
    assert "toward fake" in out["focus_summary"]

    # Magnitude map = |signed map|, and the signed map decomposes the logit
    lum = w._rgb_to_luminance(transforms.functional.to_tensor(w._resize(img))).unsqueeze(0)
    r = ex.compute_cam(model, lum, w._cam_target_layer(), method="hirescam", target_class=1)
    res = w._run_inference(lum, explain=True)
    np.testing.assert_allclose(res["heatmap"], np.abs(res["signed_map"]))
    with torch.no_grad():
        logit = model(lum)[0].item()
        const = (model.classifier.weight @ model.embedding.bias + model.classifier.bias).item()
    assert r.cam.sum() == pytest.approx(logit - const, rel=1e-3, abs=1e-4)


# ---------------------------------------------------------------------------
# Encoding and summary
# ---------------------------------------------------------------------------

def test_signed_heatmap_encoding_colors():
    signed = np.zeros((4, 4), dtype=np.float32)
    signed[0, 0], signed[3, 3] = 1.0, -1.0
    img = np.array(_decode_png(ex.signed_heatmap_to_base64(signed)))
    assert img.shape == (4, 4, 3)
    r, g, b = img[0, 0].astype(int)
    assert r > b + 50          # positive -> red
    r, g, b = img[3, 3].astype(int)
    assert b > r + 50          # negative -> blue
    assert img[1, 1].min() > 230  # zero -> near white


def test_focus_summary_has_no_semantic_face_claim():
    hm = np.zeros((100, 100), dtype=np.float32)
    hm[20:35, 40:60] = 1.0  # upper-center, concentrated: used to say "likely face"
    s = ex.compute_focus_summary(hm)
    assert "face" not in s and "subject" not in s
    assert "upper-center" in s


def test_signed_evidence_share():
    assert ex.signed_evidence_share(np.array([1.0, -1.0, 2.0])) == pytest.approx(0.75)
    assert ex.signed_evidence_share(np.zeros(3)) is None


def test_schema_accepts_new_fields():
    from app.schemas.predict import FusionMeta, PredictionResult

    r = PredictionResult(
        pred="fake", pred_int=1, prob_fake=0.7, explainability_type="signed_grad_cam",
        explainability_method="hirescam_signed", heatmap_signed_base64="abc",
    )
    assert r.heatmap_signed_base64 == "abc"
    FusionMeta(contribution_directions={"a": "fake"}, intercept=-1.0)
    # Old payloads still validate
    PredictionResult(pred="real", pred_int=0, prob_fake=0.1, explainability_type="grad_cam")
