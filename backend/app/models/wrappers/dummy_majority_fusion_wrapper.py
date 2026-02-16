"""
Wrapper for dummy majority vote fusion model.
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List

from app.core.errors import FusionError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseFusionWrapper

logger = get_logger(__name__)


class DummyMajorityFusionWrapper(BaseFusionWrapper):
    """
    Wrapper for dummy majority vote fusion models.
    
    These models are hosted on Hugging Face and contain a fusion.py
    with a predict() function that performs majority voting on submodel outputs.
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
            repo_id: Hugging Face repository ID (e.g., "DeepFakeDetector/fusion-majority-test")
            config: Configuration from config.json
            local_path: Local path where the model files are stored
        """
        super().__init__(repo_id, config, local_path)
        self._submodel_repos: List[str] = config.get("submodels", [])
        logger.info(f"Initialized DummyMajorityFusionWrapper for {repo_id}")
        logger.info(f"Submodels: {self._submodel_repos}")
    
    @property
    def submodel_repos(self) -> List[str]:
        """Get list of submodel repository IDs."""
        return self._submodel_repos
    
    def load(self) -> None:
        """
        Load the fusion predict function from the downloaded repository.
        
        Dynamically imports predict.py and extracts the predict function.
        """
        fusion_path = Path(self.local_path) / "predict.py"
        
        if not fusion_path.exists():
            raise ConfigurationError(
                message=f"predict.py not found in {self.local_path}",
                details={"repo_id": self.repo_id, "expected_path": str(fusion_path)}
            )
        
        try:
            # Create a unique module name to avoid conflicts
            module_name = f"hf_model_{self.name.replace('-', '_')}_fusion"
            
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, fusion_path)
            if spec is None or spec.loader is None:
                raise ConfigurationError(
                    message=f"Could not load spec for {fusion_path}",
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
            logger.info(f"Loaded fusion predict function from {self.repo_id}")
            
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to load fusion function from {self.repo_id}: {e}")
            raise ConfigurationError(
                message=f"Failed to load fusion model: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    def predict(
        self,
        submodel_outputs: Dict[str, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run fusion prediction on submodel outputs.
        
        Args:
            submodel_outputs: Dictionary mapping submodel name to its prediction output
            **kwargs: Additional arguments passed to the fusion function
            
        Returns:
            Standardized prediction dictionary with:
            - pred_int: 0 or 1
            - pred: "real" or "fake"
            - prob_fake: float (average of pred_ints)
            - meta: dict
        """
        if self._predict_fn is None:
            raise FusionError(
                message="Fusion model not loaded",
                details={"repo_id": self.repo_id}
            )
        
        try:
            # Call the actual fusion predict function from the HF repo
            result = self._predict_fn(submodel_outputs=submodel_outputs, **kwargs)
            
            # Validate and standardize the output
            standardized = self._standardize_output(result)
            return standardized
            
        except FusionError:
            raise
        except Exception as e:
            logger.error(f"Fusion prediction failed for {self.repo_id}: {e}")
            raise FusionError(
                message=f"Fusion prediction failed: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    def _standardize_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Standardize the fusion output to ensure consistent format.
        
        Args:
            result: Raw fusion output
            
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
