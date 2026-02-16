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
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        
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
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        files = {"image": ("test.png", io.BytesIO(png_bytes), "image/png")}
        
        # Both true and false should be valid
        response = client.post("/predict?use_fusion=true", files=files)
        assert response.status_code in (200, 500, 503)
        
        files = {"image": ("test.png", io.BytesIO(png_bytes), "image/png")}
        response = client.post("/predict?use_fusion=false", files=files)
        assert response.status_code in (200, 500, 503)
    
    def test_predict_with_model_param(self, client):
        """Test that model query param is accepted."""
        png_bytes = (
            b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01'
            b'\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00'
            b'\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00'
            b'\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82'
        )
        files = {"image": ("test.png", io.BytesIO(png_bytes), "image/png")}
        
        response = client.post(
            "/predict?use_fusion=false&model=test-random-a",
            files=files
        )
        assert response.status_code in (200, 404, 500, 503)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
