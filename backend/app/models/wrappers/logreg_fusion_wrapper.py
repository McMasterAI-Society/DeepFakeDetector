"""
Wrapper for logistic regression stacking fusion model.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

from app.core.errors import FusionError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseFusionWrapper

logger = get_logger(__name__)

# Clipping used when the fusion features are log-odds of submodel probabilities
LOGIT_EPS = 1e-6


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
        # Features the logistic regression was trained on: raw probabilities
        # ("prob", default) or log-odds ("logit").
        self._feature_transform: str = str(config.get("feature_transform", "prob")).lower()
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
    
    def _features(self, probs: List[float]) -> np.ndarray:
        """Build the [1, n] feature row the logistic regression expects."""
        X = np.asarray(probs, dtype=np.float64).reshape(1, -1)
        if self._feature_transform == "logit":
            p = np.clip(X, LOGIT_EPS, 1.0 - LOGIT_EPS)
            X = np.log(p / (1.0 - p))
        elif self._feature_transform not in ("prob", "none", "identity"):
            raise FusionError(
                message=f"Unknown feature_transform: {self._feature_transform}",
                details={"repo_id": self.repo_id}
            )
        return X

    def _linear_model(self, X: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Return (coef, intercept, features as the linear step sees them).

        Supports a bare LogisticRegression, or a scikit-learn Pipeline whose
        last step is linear (e.g. clip -> logit -> LogisticRegression): the
        earlier steps are applied to X first. If the final estimator has no
        coef_, falls back to "coef"/"intercept" from config.json, which are
        taken to be in the same feature space.
        """
        model = self._model
        if hasattr(model, "steps"):
            for _, step in model.steps[:-1]:
                if step is not None and not (isinstance(step, str) and step == "passthrough"):
                    X = step.transform(X)
            model = model.steps[-1][1]
        X = np.asarray(X, dtype=np.float64)

        if hasattr(model, "coef_"):
            coef = model.coef_
            intercept = getattr(model, "intercept_", 0.0)
        else:
            coef = self.config.get("coef", self.config.get("coefficients"))
            intercept = self.config.get("intercept", 0.0)
            if coef is None:
                raise ValueError(f"{type(model).__name__} has no coef_ and config.json has no coef")
            if isinstance(coef, dict):
                coef = [coef[name] for name in self._submodel_order]
        coef = np.asarray(coef, dtype=np.float64).reshape(-1)
        intercept = float(np.asarray(intercept, dtype=np.float64).reshape(-1)[0])
        return coef, intercept, X

    def _contribution_baseline(self, n: int) -> np.ndarray:
        """Config "contribution_baseline": list or {submodel: value}."""
        raw = self.config.get("contribution_baseline")
        if raw is None:
            return np.zeros(n)
        if isinstance(raw, dict):
            raw = [raw.get(name, 0.0) for name in self._submodel_order]
        ref = np.asarray(raw, dtype=np.float64).reshape(-1)
        if ref.shape[0] != n:
            logger.warning("contribution_baseline length mismatch; ignoring it")
            return np.zeros(n)
        return ref

    def _contributions(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Per-branch contributions to the fusion logit.

        The fusion logit is intercept + sum_i coef_i * x_i, so coef_i * x_i is
        branch i's exact additive share (positive pushes toward fake). With
        raw-probability features and positive coefficients every share is
        >= 0; set "contribution_baseline" to make shares relative to a
        reference input so that branches can also push toward real.
        Percentages normalise absolute contributions to sum to 100; the sign
        is reported separately in `contribution_directions`.
        """
        coef, intercept, Xm = self._linear_model(X)
        if coef.shape[0] != Xm.shape[1] or coef.shape[0] != len(self._submodel_order):
            raise ValueError(
                f"coef_ has {coef.shape[0]} entries for {len(self._submodel_order)} submodels"
            )
        # Optional reference point (in model feature space, e.g. training
        # means): contributions become coef_i * (x_i - ref_i) and the
        # reference's share moves into the intercept. Default: zeros.
        ref = self._contribution_baseline(Xm.shape[1])
        weighted = coef * (Xm[0] - ref)
        intercept += float((coef * ref).sum())
        total_abs = float(np.abs(weighted).sum())
        if total_abs > 0:
            pct = np.abs(weighted) / total_abs * 100.0
        else:
            pct = np.full(len(weighted), 100.0 / max(len(weighted), 1))

        names = self._submodel_order
        directions = {
            n: ("fake" if w > 0 else "real" if w < 0 else "neutral")
            for n, w in zip(names, weighted)
        }
        out = {
            "submodel_weights": {n: float(c) for n, c in zip(names, coef)},
            "weighted_contributions": {n: float(w) for n, w in zip(names, weighted)},
            "contribution_percentages": {n: float(p) for n, p in zip(names, pct)},
            "contribution_directions": directions,
            "intercept": intercept,
            "fusion_logit": float(intercept + weighted.sum()),
        }
        # Calibrated exports may divide the logit by a temperature before the
        # sigmoid; contributions stay in the uncalibrated logit space and the
        # percentages are unaffected by the scaling.
        if self.config.get("temperature") is not None:
            out["temperature"] = float(self.config["temperature"])
        return out

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
            - meta: dict with submodel probabilities, threshold and the
              per-branch contributions (submodel_weights,
              weighted_contributions, contribution_percentages,
              contribution_directions, intercept, fusion_logit)
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
            
            X = self._features(probs)
            
            # Get prediction and probability
            prob_fake = float(self._model.predict_proba(X)[0, 1])
            pred_int = 1 if prob_fake >= self._threshold else 0
            pred = "fake" if pred_int == 1 else "real"
            
            meta = {
                "submodel_probs": dict(zip(self._submodel_order, probs)),
                "threshold": self._threshold,
                "feature_transform": self._feature_transform,
            }
            try:
                meta.update(self._contributions(X))
            except Exception as e:  # never fail a prediction over its explanation
                logger.warning(f"Could not compute fusion contributions: {e}")
            
            return {
                "pred_int": pred_int,
                "pred": pred,
                "prob_fake": prob_fake,
                "meta": meta
            }
            
        except FusionError:
            raise
        except Exception as e:
            logger.error(f"Fusion prediction failed for {self.repo_id}: {e}")
            raise FusionError(
                message=f"Fusion prediction failed: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
