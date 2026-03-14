"""
Wrapper for logistic regression stacking fusion model.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

from app.core.errors import FusionError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseFusionWrapper

logger = get_logger(__name__)


class LogRegFusionWrapper(BaseFusionWrapper):
    """
    Wrapper for probability stacking fusion with logistic regression.
    
    This fusion model takes probability outputs from submodels,
    stacks them into a feature vector, and runs them through a
    trained logistic regression classifier.
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
            repo_id: Hugging Face repository ID
            config: Configuration from config.json
            local_path: Local path where the model files are stored
        """
        super().__init__(repo_id, config, local_path)
        self._model = None
        self._submodel_order: List[str] = config.get("submodel_order", [])
        self._threshold: float = config.get("threshold", 0.5)
        logger.info(f"Initialized LogRegFusionWrapper for {repo_id}")
        logger.info(f"Submodel order: {self._submodel_order}")
    
    @property
    def submodel_repos(self) -> List[str]:
        """Get list of submodel repository IDs."""
        return self.config.get("submodels", [])
    
    def load(self) -> None:
        """
        Load the logistic regression model from the downloaded repository.
        
        Loads fusion_logreg.pkl using joblib (sklearn models are saved with joblib).
        """
        model_path = Path(self.local_path) / "fusion_logreg.pkl"
        
        if not model_path.exists():
            raise ConfigurationError(
                message=f"fusion_logreg.pkl not found in {self.local_path}",
                details={"repo_id": self.repo_id, "expected_path": str(model_path)}
            )
        
        try:
            # Use joblib for sklearn models instead of pickle
            self._model = joblib.load(model_path)
            logger.info(f"Loaded logistic regression fusion model from {self.repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to load fusion model from {self.repo_id}: {e}")
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
        
        Stacks submodel probabilities in the correct order and runs
        through the logistic regression classifier.
        
        Args:
            submodel_outputs: Dictionary mapping submodel name to its prediction output
                Each output must contain "prob_fake" key
            **kwargs: Additional arguments (unused)
            
        Returns:
            Standardized prediction dictionary with:
            - pred_int: 0 or 1
            - pred: "real" or "fake"
            - prob_fake: float probability of being fake
            - meta: dict with submodel probabilities
        """
        if self._model is None:
            raise FusionError(
                message="Fusion model not loaded",
                details={"repo_id": self.repo_id}
            )
        
        try:
            # Stack submodel probabilities in the correct order
            probs = []
            for submodel_name in self._submodel_order:
                if submodel_name not in submodel_outputs:
                    raise FusionError(
                        message=f"Missing output from submodel: {submodel_name}",
                        details={
                            "repo_id": self.repo_id,
                            "missing_submodel": submodel_name,
                            "available_submodels": list(submodel_outputs.keys())
                        }
                    )
                
                output = submodel_outputs[submodel_name]
                if "prob_fake" not in output:
                    raise FusionError(
                        message=f"Submodel output missing 'prob_fake': {submodel_name}",
                        details={
                            "repo_id": self.repo_id,
                            "submodel": submodel_name,
                            "output_keys": list(output.keys())
                        }
                    )
                
                probs.append(output["prob_fake"])
            
            # Convert to numpy array and reshape for sklearn
            X = np.array(probs).reshape(1, -1)
            
            # Get prediction and probability
            prob_fake = float(self._model.predict_proba(X)[0, 1])
            pred_int = 1 if prob_fake >= self._threshold else 0
            pred = "fake" if pred_int == 1 else "real"
            
            return {
                "pred_int": pred_int,
                "pred": pred,
                "prob_fake": prob_fake,
                "meta": {
                    "submodel_probs": dict(zip(self._submodel_order, probs)),
                    "threshold": self._threshold
                }
            }
            
        except FusionError:
            raise
        except Exception as e:
            logger.error(f"Fusion prediction failed for {self.repo_id}: {e}")
            raise FusionError(
                message=f"Fusion prediction failed: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
