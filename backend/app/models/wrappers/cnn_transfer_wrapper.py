"""
Wrapper for CNN Transfer (EfficientNet-B0) submodel.
"""

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from app.core.errors import InferenceError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseSubmodelWrapper
from app.services.explainability import (
    CAM_METHODS,
    compute_cam,
    compute_focus_summary,
    heatmap_to_base64,
    predicted_class,
)

logger = get_logger(__name__)


class CNNTransferWrapper(BaseSubmodelWrapper):
    """
    Wrapper for CNN Transfer model using EfficientNet-B0 backbone.
    
    Model expects 224x224 RGB images with ImageNet normalization.
    """
    
    def __init__(
        self,
        repo_id: str,
        config: Dict[str, Any],
        local_path: str
    ):
        super().__init__(repo_id, config, local_path)
        self._model: Optional[nn.Module] = None
        self._transform: Optional[transforms.Compose] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._threshold = config.get("threshold", 0.5)
        logger.info(f"Initialized CNNTransferWrapper for {repo_id}")
    
    def load(self) -> None:
        """Load the EfficientNet-B0 model with trained weights."""
        weights_path = Path(self.local_path) / "model.pth"
        preprocess_path = Path(self.local_path) / "preprocess.json"
        
        if not weights_path.exists():
            raise ConfigurationError(
                message=f"model.pth not found in {self.local_path}",
                details={"repo_id": self.repo_id, "expected_path": str(weights_path)}
            )
        
        try:
            # Load preprocessing config
            preprocess_config = {}
            if preprocess_path.exists():
                with open(preprocess_path, "r") as f:
                    preprocess_config = json.load(f)
            
            # Build transform pipeline
            input_size = preprocess_config.get("input_size", [224, 224])
            if isinstance(input_size, int):
                input_size = [input_size, input_size]
            
            normalize_config = preprocess_config.get("normalize", {})
            mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
            std = normalize_config.get("std", [0.229, 0.224, 0.225])
            
            self._transform = transforms.Compose([
                transforms.Resize(input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            # Create model architecture
            num_classes = self.config.get("num_classes", 2)
            self._model = efficientnet_b0(weights=None)
            
            # Replace classifier for binary classification
            in_features = self._model.classifier[1].in_features
            self._model.classifier = nn.Sequential(
                nn.Dropout(p=0.2, inplace=True),
                nn.Linear(in_features, num_classes)
            )
            
            # Load trained weights
            state_dict = torch.load(weights_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            
            # Mark as loaded
            self._predict_fn = self._run_inference
            logger.info(f"Loaded CNN Transfer model from {self.repo_id}")
            
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to load CNN Transfer model: {e}")
            raise ConfigurationError(
                message=f"Failed to load model: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    # Default CAM target: features[5] (112 ch, 14x14 at 224 input). features[-2]
    # and features[-3] are still 7x7 in EfficientNet-B0, so they add no spatial
    # detail over features[-1]. Override with config "cam_layer" / "cam_method".
    DEFAULT_CAM_LAYER = 5
    DEFAULT_CAM_METHOD = "hirescam"

    def _cam_settings(self) -> Tuple[nn.Module, str]:
        layer_idx = int(self.config.get("cam_layer", self.DEFAULT_CAM_LAYER))
        method = str(self.config.get("cam_method", self.DEFAULT_CAM_METHOD)).lower()
        if method not in CAM_METHODS:
            logger.warning(f"Unknown cam_method {method!r}; using {self.DEFAULT_CAM_METHOD}")
            method = self.DEFAULT_CAM_METHOD
        return self._model.features[layer_idx], method

    def _run_inference(
        self,
        image_tensor: torch.Tensor,
        explain: bool = False
    ) -> Dict[str, Any]:
        """Run model inference on preprocessed tensor."""
        heatmap = None
        method = None

        if explain:
            # Logits and CAM from a single forward + partial backward pass
            target_layer, method = self._cam_settings()
            cam = compute_cam(
                self._model,
                image_tensor,
                target_layer,
                method=method,
                target_class=lambda lg: predicted_class(lg, self._threshold),
                output_size=(224, 224),
            )
            logits = cam.logits
            heatmap = cam.heatmap
        else:
            with torch.no_grad():
                logits = self._model(image_tensor)

        probs = F.softmax(logits, dim=1)
        prob_fake = probs[0, 1].item()
        pred_int = 1 if prob_fake >= self._threshold else 0

        result = {
            "logits": logits[0].detach().cpu().numpy().tolist(),
            "prob_fake": prob_fake,
            "pred_int": pred_int
        }

        if heatmap is not None:
            result["heatmap"] = heatmap
            result["explainability_method"] = method

        return result

    def predict(
        self,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        explain: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run prediction on an image.
        
        Args:
            image: PIL Image object
            image_bytes: Raw image bytes (will be converted to PIL Image)
            explain: If True, compute GradCAM heatmap
            
        Returns:
            Standardized prediction dictionary with optional heatmap
        """
        if self._model is None or self._transform is None:
            raise InferenceError(
                message="Model not loaded",
                details={"repo_id": self.repo_id}
            )
        
        try:
            # Convert bytes to PIL Image if needed
            if image is None and image_bytes is not None:
                import io
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            elif image is not None:
                image = image.convert("RGB")
            else:
                raise InferenceError(
                    message="No image provided",
                    details={"repo_id": self.repo_id}
                )
            
            # Preprocess
            image_tensor = self._transform(image).unsqueeze(0).to(self._device)
            
            # Run inference
            result = self._run_inference(image_tensor, explain=explain)
            
            # Standardize output
            labels = self.config.get("labels", {"0": "real", "1": "fake"})
            pred_int = result["pred_int"]
            
            output = {
                "pred_int": pred_int,
                "pred": labels.get(str(pred_int), "unknown"),
                "prob_fake": result["prob_fake"],
                "meta": {
                    "model": self.name,
                    "threshold": self._threshold,
                    "logits": result["logits"]
                }
            }
            
            # Add heatmap if requested
            if explain and "heatmap" in result:
                heatmap = result["heatmap"]
                output["heatmap_base64"] = heatmap_to_base64(heatmap)
                output["explainability_type"] = "grad_cam"
                output["explainability_method"] = result.get("explainability_method")
                output["focus_summary"] = compute_focus_summary(heatmap)
            
            return output
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {self.repo_id}: {e}")
            raise InferenceError(
                message=f"Prediction failed: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
