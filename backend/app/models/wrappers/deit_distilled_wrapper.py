"""
Wrapper for DeiT Distilled submodel.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image
from torchvision import transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

from app.core.errors import InferenceError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseSubmodelWrapper

logger = get_logger(__name__)


def create_custom_mlp_head(in_features: int = 768, num_classes: int = 2) -> nn.Sequential:
    """
    Create custom MLP head for DeiT model matching training configuration.
    
    Returns nn.Sequential to match saved state dict keys (0, 1, 4 indices).
    """
    return nn.Sequential(
        nn.LayerNorm(in_features),   # 0
        nn.Linear(in_features, 512), # 1
        nn.GELU(),                   # 2 (no params)
        nn.Dropout(p=0.2),           # 3 (no params)
        nn.Linear(512, num_classes)  # 4
    )


class DeiTDistilledWrapper(BaseSubmodelWrapper):
    """
    Wrapper for DeiT Distilled model.
    
    Model expects 224x224 RGB images with ImageNet normalization.
    Uses a custom MLP head for classification.
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
        logger.info(f"Initialized DeiTDistilledWrapper for {repo_id}")
    
    def load(self) -> None:
        """Load the DeiT model with custom head and trained weights."""
        if not TIMM_AVAILABLE:
            raise ConfigurationError(
                message="timm package not installed. Run: pip install timm",
                details={"repo_id": self.repo_id}
            )
        
        weights_path = Path(self.local_path) / "deit_distilled_custom_head.pt"
        preprocess_path = Path(self.local_path) / "preprocess.json"
        
        if not weights_path.exists():
            raise ConfigurationError(
                message=f"deit_distilled_custom_head.pt not found in {self.local_path}",
                details={"repo_id": self.repo_id, "expected_path": str(weights_path)}
            )
        
        try:
            # Load preprocessing config
            preprocess_config = {}
            if preprocess_path.exists():
                with open(preprocess_path, "r") as f:
                    preprocess_config = json.load(f)
            
            # Build transform pipeline
            input_size = preprocess_config.get("input_size", 224)
            if isinstance(input_size, list):
                input_size = input_size[0]
            
            normalize_config = preprocess_config.get("normalize", {})
            mean = normalize_config.get("mean", [0.485, 0.456, 0.406])
            std = normalize_config.get("std", [0.229, 0.224, 0.225])
            
            # Use bicubic interpolation as specified
            interpolation = preprocess_config.get("interpolation", "bicubic")
            interp_mode = transforms.InterpolationMode.BICUBIC if interpolation == "bicubic" else transforms.InterpolationMode.BILINEAR
            
            self._transform = transforms.Compose([
                transforms.Resize((input_size, input_size), interpolation=interp_mode),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            # Create model architecture
            model_name = self.config.get("model_name", "deit_base_distilled_patch16_224")
            num_classes = self.config.get("num_classes", 2)
            
            # Create base model without pretrained weights
            self._model = timm.create_model(model_name, pretrained=False, num_classes=0)
            
            # Replace heads with custom MLP heads (Sequential assigned directly)
            # Note: state dict has separate keys for head and head_dist, so don't share
            hidden_dim = 768  # DeiT base hidden dimension
            self._model.head = create_custom_mlp_head(hidden_dim, num_classes)
            self._model.head_dist = create_custom_mlp_head(hidden_dim, num_classes)
            
            # Load trained weights
            state_dict = torch.load(weights_path, map_location=self._device, weights_only=True)
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            
            # Mark as loaded
            self._predict_fn = self._run_inference
            logger.info(f"Loaded DeiT Distilled model from {self.repo_id}")
            
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to load DeiT Distilled model: {e}")
            raise ConfigurationError(
                message=f"Failed to load model: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    def _run_inference(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run model inference on preprocessed tensor."""
        with torch.no_grad():
            # In eval mode, DeiT returns single tensor
            logits = self._model(image_tensor)
            probs = F.softmax(logits, dim=1)
            prob_fake = probs[0, 1].item()  # Index 1 is "fake"
            pred_int = 1 if prob_fake >= self._threshold else 0
            
        return {
            "logits": logits[0].cpu().numpy().tolist(),
            "prob_fake": prob_fake,
            "pred_int": pred_int
        }
    
    def predict(
        self,
        image: Optional[Image.Image] = None,
        image_bytes: Optional[bytes] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run prediction on an image.
        
        Args:
            image: PIL Image object
            image_bytes: Raw image bytes (will be converted to PIL Image)
            
        Returns:
            Standardized prediction dictionary
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
            result = self._run_inference(image_tensor)
            
            # Standardize output
            class_mapping = self.config.get("class_mapping", {"0": "real", "1": "fake"})
            pred_int = result["pred_int"]
            
            return {
                "pred_int": pred_int,
                "pred": class_mapping.get(str(pred_int), "unknown"),
                "prob_fake": result["prob_fake"],
                "meta": {
                    "model": self.name,
                    "threshold": self._threshold,
                    "logits": result["logits"]
                }
            }
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {self.repo_id}: {e}")
            raise InferenceError(
                message=f"Prediction failed: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
