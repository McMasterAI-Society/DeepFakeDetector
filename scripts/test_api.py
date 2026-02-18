"""
API endpoint test script.

Tests all endpoints of the DeepFake Detector API.
Assumes the server is running at http://localhost:8000

Usage:
    python scripts/test_api.py
"""

import sys
import time
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
TEST_IMAGE = Path(__file__).parent.parent / "backend" / "testimages" / "image.png"


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, passed: bool, details: str = ""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}: {name}")
    if details:
        print(f"         {details}")


def test_health():
    """Test /health endpoint."""
    print_header("Testing /health")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        passed = resp.status_code == 200 and resp.json().get("status") == "ok"
        print_result("/health", passed, f"status={resp.json().get('status')}")
        return passed
    except Exception as e:
        print_result("/health", False, str(e))
        return False


def test_ready():
    """Test /ready endpoint."""
    print_header("Testing /ready")
    try:
        resp = requests.get(f"{BASE_URL}/ready", timeout=5)
        data = resp.json()
        passed = resp.status_code == 200 and data.get("models_loaded", False)
        print_result("/ready", passed, f"models_loaded={data.get('models_loaded')}")
        if data.get("submodels"):
            print(f"         Submodels: {data['submodels']}")
        return passed
    except Exception as e:
        print_result("/ready", False, str(e))
        return False


def test_models():
    """Test /models endpoint."""
    print_header("Testing /models")
    try:
        resp = requests.get(f"{BASE_URL}/models", timeout=5)
        data = resp.json()
        passed = resp.status_code == 200 and "submodels" in data
        print_result("/models", passed, f"total_count={data.get('total_count')}")
        
        if data.get("fusion"):
            print(f"         Fusion: {data['fusion'].get('name')}")
        
        submodel_names = [s.get("name") for s in data.get("submodels", [])]
        if submodel_names:
            print(f"         Submodels: {submodel_names}")
        
        return passed, submodel_names
    except Exception as e:
        print_result("/models", False, str(e))
        return False, []


def test_predict_fusion(image_path: Path):
    """Test /predict with fusion (default)."""
    print_header("Testing /predict (fusion=true)")
    try:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            resp = requests.post(
                f"{BASE_URL}/predict",
                files=files,
                params={"use_fusion": True, "return_submodels": True},
                timeout=60
            )
        
        data = resp.json()
        passed = resp.status_code == 200 and "prediction" in data
        
        if passed:
            pred = data["prediction"]
            print_result("/predict (fusion)", True, 
                        f"pred={pred.get('pred')}, confidence={pred.get('confidence'):.3f}")
            
            # Show submodel results
            if data.get("submodel_results"):
                print("         Submodel breakdown:")
                for sub in data["submodel_results"]:
                    print(f"           - {sub.get('model')}: {sub.get('pred')} (prob={sub.get('prob_fake', 'N/A')})")
            
            # Show timing
            if data.get("timing"):
                print(f"         Timing: total={data['timing'].get('total_ms', 0):.0f}ms")
        else:
            print_result("/predict (fusion)", False, data.get("detail", str(data)))
        
        return passed
    except Exception as e:
        print_result("/predict (fusion)", False, str(e))
        return False


def test_predict_single_submodel(image_path: Path, model_name: str):
    """Test /predict with a specific submodel."""
    try:
        with open(image_path, "rb") as f:
            files = {"image": (image_path.name, f, "image/png")}
            resp = requests.post(
                f"{BASE_URL}/predict",
                files=files,
                params={"use_fusion": False, "model": model_name},
                timeout=60
            )
        
        data = resp.json()
        passed = resp.status_code == 200 and "prediction" in data
        
        if passed:
            pred = data["prediction"]
            details = f"pred={pred.get('pred')}, prob_fake={pred.get('prob_fake', 'N/A')}"
        else:
            details = data.get("detail", str(data))
        
        print_result(f"/predict (model={model_name})", passed, details)
        return passed
    except Exception as e:
        print_result(f"/predict (model={model_name})", False, str(e))
        return False


def test_all_submodels(image_path: Path, submodel_names: list):
    """Test each submodel individually."""
    print_header("Testing /predict (individual submodels)")
    
    results = []
    for name in submodel_names:
        results.append(test_predict_single_submodel(image_path, name))
    
    return all(results) if results else True


def main():
    print("\n" + "="*60)
    print("  DeepFake Detector API Test Suite")
    print("="*60)
    
    # Check test image exists
    if not TEST_IMAGE.exists():
        print(f"\n  ERROR: Test image not found: {TEST_IMAGE}")
        print("  Please ensure the test image exists.")
        sys.exit(1)
    
    print(f"\n  Test image: {TEST_IMAGE}")
    print(f"  Server: {BASE_URL}")
    
    # Check server is running
    print("\n  Checking server connectivity...")
    try:
        requests.get(f"{BASE_URL}/health", timeout=3)
    except requests.exceptions.ConnectionError:
        print(f"\n  ERROR: Cannot connect to server at {BASE_URL}")
        print("  Start the server with: cd backend && python -m app.main")
        sys.exit(1)
    
    print("  Server is reachable!\n")
    
    # Run tests
    results = []
    
    # Health check
    results.append(test_health())
    
    # Readiness check
    results.append(test_ready())
    
    # List models
    models_passed, submodel_names = test_models()
    results.append(models_passed)
    
    # Predict with fusion
    results.append(test_predict_fusion(TEST_IMAGE))
    
    # Test each submodel
    if submodel_names:
        results.append(test_all_submodels(TEST_IMAGE, submodel_names))
    
    # Summary
    print_header("Summary")
    passed = sum(results)
    total = len(results)
    
    if all(results):
        print(f"  All {total} test groups passed!")
    else:
        print(f"  {passed}/{total} test groups passed")
    
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
