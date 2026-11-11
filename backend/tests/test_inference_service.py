"""
Tests for InferenceService: single image decode, grad context, warm-up.

Submodel wrappers are mocked, so no model weights are needed.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from app.core.errors import ImageProcessingError
from app.services import inference_service as inference_module
from app.services.inference_service import InferenceService, decode_image


def _png_bytes(mode: str = "RGB", size=(32, 24)) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color=128 if mode == "L" else (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


class _RecordingSubmodel:
    """Fake wrapper recording what it receives and the grad state it ran in."""

    def __init__(self, name: str):
        self.name = name
        self.calls = []

    def predict(self, image=None, image_bytes=None, explain=False, **kwargs):
        self.calls.append(
            {
                "image": image,
                "image_bytes": image_bytes,
                "explain": explain,
                "inference_mode": torch.is_inference_mode_enabled(),
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        return {"pred_int": 0, "pred": "real", "prob_fake": 0.1, "meta": {}}


def _service_with(submodels, is_loaded=True) -> InferenceService:
    registry = MagicMock()
    registry.get_all_submodels.return_value = submodels
    registry.get_submodel.side_effect = lambda key: submodels[key]
    registry.is_loaded = is_loaded
    with patch.object(inference_module, "get_model_registry", return_value=registry):
        return InferenceService()


def _models():
    return {name: _RecordingSubmodel(name) for name in ("cnn", "vit", "deit", "gradfield")}


class TestDecodeOnce:
    def test_predict_all_decodes_once_and_shares_image(self):
        models = _models()
        service = _service_with(models)
        data = _png_bytes()

        with patch.object(
            inference_module.Image, "open", wraps=inference_module.Image.open
        ) as open_spy:
            results = service.predict_all_submodels(image_bytes=data, explain=False)

        assert open_spy.call_count == 1
        assert set(results) == set(models)
        images = [m.calls[0]["image"] for m in models.values()]
        assert all(isinstance(img, Image.Image) for img in images)
        assert all(img is images[0] for img in images), "all wrappers must get the same decoded image"
        assert images[0].mode == "RGB"
        # bytes still forwarded for bytes-only wrappers
        assert all(m.calls[0]["image_bytes"] is data for m in models.values())

    def test_decode_converts_to_rgb_like_wrappers(self):
        data = _png_bytes(mode="L")
        decoded = decode_image(image_bytes=data)
        expected = Image.open(io.BytesIO(data)).convert("RGB")
        assert decoded.mode == "RGB"
        assert decoded.tobytes() == expected.tobytes()

    def test_decode_passes_through_rgb_image(self):
        img = Image.new("RGB", (8, 8))
        assert decode_image(image=img) is img

    def test_invalid_bytes_raise_image_processing_error(self):
        service = _service_with(_models())
        with pytest.raises(ImageProcessingError):
            service.predict_all_submodels(image_bytes=b"not an image")

    def test_predict_single_decodes(self):
        models = _models()
        service = _service_with(models)
        service.predict_single("vit", image_bytes=_png_bytes())
        assert isinstance(models["vit"].calls[0]["image"], Image.Image)


class TestGradContext:
    def test_non_explain_runs_under_inference_mode(self):
        models = _models()
        service = _service_with(models)
        service.predict_all_submodels(image_bytes=_png_bytes(), explain=False)
        for m in models.values():
            assert m.calls[0]["inference_mode"] is True
            assert m.calls[0]["explain"] is False

    def test_explain_keeps_autograd(self):
        models = _models()
        service = _service_with(models)
        service.predict_all_submodels(image_bytes=_png_bytes(), explain=True)
        for m in models.values():
            assert m.calls[0]["inference_mode"] is False
            assert m.calls[0]["grad_enabled"] is True
            assert m.calls[0]["explain"] is True

    def test_fp16_off_by_default(self):
        assert inference_module.settings.INFERENCE_FP16 is False


class TestWarmup:
    def test_warm_up_runs_each_submodel_once_without_explain(self):
        models = _models()
        service = _service_with(models)
        durations = service.warm_up()
        assert set(durations) == set(models)
        for m in models.values():
            assert len(m.calls) == 1
            assert m.calls[0]["explain"] is False
            assert m.calls[0]["inference_mode"] is True

    def test_warm_up_skips_failing_model(self):
        models = _models()
        models["vit"].predict = MagicMock(side_effect=RuntimeError("boom"))
        service = _service_with(models)
        durations = service.warm_up()
        assert "vit" not in durations
        assert len(durations) == 3

    def test_warmup_can_be_disabled(self):
        service = MagicMock()
        registry = MagicMock(is_loaded=True)
        with patch.object(inference_module.settings, "WARMUP_ON_STARTUP", False), \
             patch.object(inference_module, "get_inference_service", return_value=service), \
             patch.object(inference_module, "get_model_registry", return_value=registry):
            assert inference_module.warm_up_on_startup() is None
        service.warm_up.assert_not_called()

    def test_warmup_enabled_by_default(self):
        service = MagicMock()
        service.warm_up.return_value = {"cnn": 0.01}
        registry = MagicMock(is_loaded=True)
        assert inference_module.settings.WARMUP_ON_STARTUP is True
        with patch.object(inference_module, "get_inference_service", return_value=service), \
             patch.object(inference_module, "get_model_registry", return_value=registry):
            assert inference_module.warm_up_on_startup() == {"cnn": 0.01}
        service.warm_up.assert_called_once()

    def test_warmup_skipped_when_models_not_loaded(self):
        service = MagicMock()
        registry = MagicMock(is_loaded=False)
        with patch.object(inference_module, "get_inference_service", return_value=service), \
             patch.object(inference_module, "get_model_registry", return_value=registry):
            assert inference_module.warm_up_on_startup() is None
        service.warm_up.assert_not_called()
