"""
Tests for API endpoints.

These tests use FastAPI TestClient to test endpoints.
Note: For full integration tests, models need to be available.
      These tests focus on endpoint structure and basic behavior.
"""

import io
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

# Mock the model loading before importing the app
with patch('app.services.model_registry.ModelRegistry.load_from_fusion_repo', new_callable=AsyncMock):
    from app.main import app


def _png(w, h, color=(120, 80, 40)):
    from PIL import Image
    b = io.BytesIO(); Image.new("RGB", (w, h), color).save(b, format="PNG"); return b.getvalue()


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""
    
    def test_health_returns_ok(self, client):
        """Test that /health returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "manifest_hash" in data
        assert isinstance(data.get("models"), list)
    
    def test_ready_returns_response(self, client):
        """Test that /ready returns a valid response structure."""
        response = client.get("/ready")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert data["status"] in ("ready", "not_ready")


class TestModelsEndpoint:
    """Tests for models listing endpoint."""
    
    def test_models_returns_list(self, client):
        """Test that /models returns model information."""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "fusion" in data
        assert "submodels" in data
        assert "total_count" in data
        assert "manifest_hash" in data
        assert isinstance(data["submodels"], list)


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_api_info(self, client):
        """Test that / returns API information."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "docs" in data


class TestPredictEndpoint:
    """Tests for predict endpoint structure."""
    
    def test_predict_requires_image(self, client):
        """Test that /predict requires an image file."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error - missing required field
    
    def test_predict_accepts_file_upload(self, client):
        """Test that /predict accepts multipart file upload format."""
        # Create a minimal valid PNG image (1x1 pixel)
        png_bytes = _png(128, 128)
        
        # This will fail if models aren't loaded, but we're testing the route structure
        files = {"image": ("test.png", io.BytesIO(png_bytes), "image/png")}
        response = client.post("/predict", files=files)
        
        # Either 200 (models loaded) or 503 (models not loaded) is acceptable
        # 422 would indicate wrong request format
        assert response.status_code in (200, 500, 503)


class TestQueryParameters:
    """Tests for query parameter handling."""
    
    def test_predict_with_use_fusion_param(self, client):
        """Test that use_fusion query param is accepted."""
        png_bytes = _png(128, 128)
        files = {"image": ("test.png", io.BytesIO(png_bytes), "image/png")}
        
        # Both true and false should be valid
        response = client.post("/predict?use_fusion=true", files=files)
        assert response.status_code in (200, 500, 503)
        
        files = {"image": ("test.png", io.BytesIO(png_bytes), "image/png")}
        response = client.post("/predict?use_fusion=false", files=files)
        assert response.status_code in (200, 500, 503)
    
    def test_predict_with_model_param(self, client):
        """Test that model query param is accepted."""
        png_bytes = _png(128, 128)
        files = {"image": ("test.png", io.BytesIO(png_bytes), "image/png")}
        
        response = client.post(
            "/predict?use_fusion=false&model=test-random-a",
            files=files
        )
        assert response.status_code in (200, 404, 500, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestUploadValidation:
    """Audit 11: invalid uploads get clean 4xx errors, never 500s or meaningless verdicts."""

    def post(self, client, data, name="x.png", mime="image/png"):
        return client.post("/predict", files={"image": (name, io.BytesIO(data), mime)})

    def test_tiny_image_rejected(self, client):
        r = self.post(client, _png(32, 24))
        assert r.status_code == 400 and "too small" in r.text

    def test_truncated_image_rejected(self, client):
        b = _png(256, 256)
        from PIL import Image
        jb = io.BytesIO(); Image.new("RGB", (512, 512), (10, 200, 30)).save(jb, format="JPEG", quality=90)
        r = self.post(client, jb.getvalue()[: len(jb.getvalue()) // 3], "x.jpg", "image/jpeg")
        assert r.status_code == 400 and ("corrupt" in r.text or "truncated" in r.text or "Could not read" in r.text)

    def test_pixel_bomb_rejected_with_413(self, client, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "MAX_IMAGE_PIXELS", 100 * 100)
        r = self.post(client, _png(200, 200))
        assert r.status_code == 413 and "too large" in r.text


class TestExplainModelErrors:
    def test_llm_quota_maps_to_503(self, client, monkeypatch):
        import app.api.routes_predict as rp
        from app.core.errors import LLMUnavailableError

        class FakeLLM:
            enabled = True

            def generate_single_model_explanation(self, **kw):
                raise LLMUnavailableError(message="402 RESOURCE_EXHAUSTED prepayment credits depleted")

        monkeypatch.setattr(rp, "get_llm_service", lambda: FakeLLM())
        r = client.post("/explain-model", data={"model_name": "clip-b16", "prob_fake": "0.7"},
                        files={"image": ("x.png", io.BytesIO(_png(128, 128)), "image/png")})
        assert r.status_code == 503 and "temporarily unavailable" in r.text
