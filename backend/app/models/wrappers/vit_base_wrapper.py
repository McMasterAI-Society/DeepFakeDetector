"""
Wrapper for ViT Base submodel.
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


class ViTWithMLPHead(nn.Module):
    """
    ViT model wrapper matching the training checkpoint format.
    
    The checkpoint was saved with:
    - self.vit = timm ViT backbone (num_classes=0)
    - self.fc1 = Linear(768, hidden)
    - self.fc2 = Linear(hidden, num_classes)
    """
    
    def __init__(self, arch: str = "vit_base_patch16_224", num_classes: int = 2, hidden_dim: int = 512):
        super().__init__()
        # Create backbone without classification head
        self.vit = timm.create_model(arch, pretrained=False, num_classes=0)
        embed_dim = self.vit.embed_dim  # 768 for ViT-Base
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.vit(x)  # [B, embed_dim]
        x = F.relu(self.fc1(features))
        logits = self.fc2(x)
        return logits


class ViTBaseWrapper(BaseSubmodelWrapper):
    """
    Wrapper for ViT Base model (Vision Transformer).
    
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
        logger.info(f"Initialized ViTBaseWrapper for {repo_id}")
    
    def load(self) -> None:
        """Load the ViT Base model with trained weights."""
        if not TIMM_AVAILABLE:
            raise ConfigurationError(
                message="timm package not installed. Run: pip install timm",
                details={"repo_id": self.repo_id}
            )
        
        weights_path = Path(self.local_path) / "deepfake_vit_finetuned_wildfake.pth"
        preprocess_path = Path(self.local_path) / "preprocess.json"
        
        if not weights_path.exists():
            raise ConfigurationError(
                message=f"deepfake_vit_finetuned_wildfake.pth not found in {self.local_path}",
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
            
            # Create model architecture matching the training checkpoint format
            arch = self.config.get("arch", "vit_base_patch16_224")
            num_classes = self.config.get("num_classes", 2)
            # MLP hidden dim is 512 per training notebook (fc1: 768->512, fc2: 512->2)
            # Note: config.hidden_dim (768) is ViT embedding dim, not MLP hidden dim
            mlp_hidden_dim = self.config.get("mlp_hidden_dim", 512)
            
            # Use custom wrapper that matches checkpoint structure (vit.* + fc1/fc2)
            self._model = ViTWithMLPHead(arch=arch, num_classes=num_classes, hidden_dim=mlp_hidden_dim)
            
            # Load trained weights
            checkpoint = torch.load(weights_path, map_location=self._device, weights_only=False)
            
            # Handle training checkpoint format (has "model", "optimizer_state", "epoch" keys)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            
            # Mark as loaded
            self._predict_fn = self._run_inference
            logger.info(f"Loaded ViT Base model from {self.repo_id}")
            
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to load ViT Base model: {e}")
            raise ConfigurationError(
                message=f"Failed to load model: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    def _run_inference(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        """Run model inference on preprocessed tensor."""
        with torch.no_grad():
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
            labels = self.config.get("labels", {"0": "real", "1": "fake"})
            pred_int = result["pred_int"]
            
            return {
                "pred_int": pred_int,
                "pred": labels.get(str(pred_int), "unknown"),
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
