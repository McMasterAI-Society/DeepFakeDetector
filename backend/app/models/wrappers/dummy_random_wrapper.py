"""
Wrapper for dummy random submodels.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image

from app.core.errors import InferenceError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseSubmodelWrapper

logger = get_logger(__name__)


class DummyRandomWrapper(BaseSubmodelWrapper):
    """
    Wrapper for dummy random prediction models.
    
    These models are hosted on Hugging Face and contain a predict.py
    with a predict() function that returns random predictions.
    """
    
    def __init__(
        self,
        repo_id: str,
        config: Dict[str, Any],
        local_path: str
    ):
        """
        Initialize the wrapper.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "DeepFakeDetector/test-random-a")
            config: Configuration from config.json
            local_path: Local path where the model files are stored
        """
        super().__init__(repo_id, config, local_path)
        logger.info(f"Initialized DummyRandomWrapper for {repo_id}")
    
    def load(self) -> None:
        """
        Load the predict function from the downloaded repository.
        
        Dynamically imports predict.py and extracts the predict function.
        """
        predict_path = Path(self.local_path) / "predict.py"
        
        if not predict_path.exists():
            raise ConfigurationError(
                message=f"predict.py not found in {self.local_path}",
                details={"repo_id": self.repo_id, "expected_path": str(predict_path)}
            )
        
        try:
            # Create a unique module name to avoid conflicts
            module_name = f"hf_model_{self.name.replace('-', '_')}_predict"
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, predict_path)
            if spec is None or spec.loader is None:
                raise ConfigurationError(
                    message=f"Could not load spec for {predict_path}",
                    details={"repo_id": self.repo_id}
                )
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Get the predict function
            if not hasattr(module, "predict"):
                raise ConfigurationError(
                    message=f"predict.py does not have a 'predict' function",
                    details={"repo_id": self.repo_id}
                )
            
            self._predict_fn = module.predict
            logger.info(f"Loaded predict function from {self.repo_id}")
            
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to load predict function from {self.repo_id}: {e}")
            raise ConfigurationError(
                message=f"Failed to load model: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    def predict(
        self,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run prediction on an image.
        
        Args:
            image: PIL Image object (optional for dummy model)
            image_bytes: Raw image bytes (optional for dummy model)
            **kwargs: Additional arguments passed to the model
            
        Returns:
            Standardized prediction dictionary with:
            - pred_int: 0 or 1
            - pred: "real" or "fake"
            - prob_fake: float
            - meta: dict
        """
        if self._predict_fn is None:
            raise InferenceError(
                message="Model not loaded",
                details={"repo_id": self.repo_id}
            )
        
        try:
            # Call the actual predict function from the HF repo
            result = self._predict_fn(image_bytes=image_bytes, **kwargs)
            
            # Validate and standardize the output
            standardized = self._standardize_output(result)
            return standardized
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {self.repo_id}: {e}")
            raise InferenceError(
                message=f"Prediction failed: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    def _standardize_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize the model output to ensure consistent format.
        
        Args:
            result: Raw model output
            
        Returns:
            Standardized dictionary
        """
        pred_int = result.get("pred_int", 0)
        
        # Ensure pred_int is 0 or 1
        if pred_int not in (0, 1):
            pred_int = 1 if pred_int > 0.5 else 0
        
        # Generate pred label if not present
        pred = result.get("pred")
        if pred is None:
            pred = "fake" if pred_int == 1 else "real"
        
        # Generate prob_fake if not present
        prob_fake = result.get("prob_fake")
        if prob_fake is None:
            prob_fake = float(pred_int)
        
        return {
            "pred_int": pred_int,
            "pred": pred,
            "prob_fake": float(prob_fake),
            "meta": result.get("meta", {})
        }
