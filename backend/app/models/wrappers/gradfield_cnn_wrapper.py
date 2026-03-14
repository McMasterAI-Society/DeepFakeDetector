"""
Wrapper for Gradient Field CNN submodel.
"""

import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from PIL import Image
from torchvision import transforms

from app.core.errors import InferenceError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseSubmodelWrapper
from app.services.explainability import heatmap_to_base64, compute_focus_summary

logger = get_logger(__name__)


class CompactGradientNet(nn.Module):
    """
    CNN for gradient field classification with discriminative features.
    
    Input: Luminance image (1-channel)
    Internal: Computes 6-channel gradient field [luminance, Gx, Gy, magnitude, angle, coherence]
    Output: Logits and embeddings
    """
    
    def __init__(self, depth=4, base_filters=32, dropout=0.3, embedding_dim=128):
        super().__init__()
        
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        
        # Gaussian kernel for structure tensor smoothing
        gaussian = torch.tensor([[1, 4, 6, 4, 1], [4, 16, 24, 16, 4],
                                 [6, 24, 36, 24, 6], [4, 16, 24, 16, 4],
                                 [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
        self.register_buffer('gaussian', gaussian.view(1, 1, 5, 5))
        
        # Input normalization and channel mixing
        self.input_norm = nn.BatchNorm2d(6)
        self.channel_mix = nn.Sequential(
            nn.Conv2d(6, 6, kernel_size=1),
            nn.ReLU()
        )
        
        # CNN layers
        layers = []
        in_ch = 6
        for i in range(depth):
            out_ch = base_filters * (2**i)
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ])
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.embedding = nn.Linear(out_ch, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 1)
    
    def compute_gradient_field(self, luminance):
        """Compute 6-channel gradient field on GPU (includes luminance)."""
        G_x = F.conv2d(luminance, self.sobel_x, padding=1)
        G_y = F.conv2d(luminance, self.sobel_y, padding=1)
        
        magnitude = torch.sqrt(G_x**2 + G_y**2 + 1e-8)
        angle = torch.atan2(G_y, G_x) / math.pi
        
        # Structure tensor for coherence
        Gxx, Gxy, Gyy = G_x * G_x, G_x * G_y, G_y * G_y
        Sxx = F.conv2d(Gxx, self.gaussian, padding=2)
        Sxy = F.conv2d(Gxy, self.gaussian, padding=2)
        Syy = F.conv2d(Gyy, self.gaussian, padding=2)
        
        trace = Sxx + Syy
        det_term = torch.sqrt((Sxx - Syy)**2 + 4 * Sxy**2 + 1e-8)
        lambda1, lambda2 = 0.5 * (trace + det_term), 0.5 * (trace - det_term)
        coherence = ((lambda1 - lambda2) / (lambda1 + lambda2 + 1e-8))**2
        
        magnitude_scaled = torch.log1p(magnitude * 10)
        
        return torch.cat([luminance, G_x, G_y, magnitude_scaled, angle, coherence], dim=1)
    
    def forward(self, luminance):
        x = self.compute_gradient_field(luminance)
        x = self.input_norm(x)
        x = self.channel_mix(x)
        x = self.cnn(x)
        x = self.global_pool(x).flatten(1)
        emb = self.embedding(x)
        logit = self.classifier(emb)
        return logit.squeeze(1), emb


class GradfieldCNNWrapper(BaseSubmodelWrapper):
    """
    Wrapper for Gradient Field CNN model.
    
    Model expects 256x256 luminance images.
    Internally computes Sobel gradients and other discriminative features.
    """
    
    # BT.709 luminance coefficients
    R_COEFF = 0.2126
    G_COEFF = 0.7152
    B_COEFF = 0.0722
    
    def __init__(
        self,
        repo_id: str,
        config: Dict[str, Any],
        local_path: str
    ):
        super().__init__(repo_id, config, local_path)
        self._model: Optional[nn.Module] = None
        self._resize: Optional[transforms.Resize] = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._threshold = config.get("threshold", 0.5)
        logger.info(f"Initialized GradfieldCNNWrapper for {repo_id}")
    
    def load(self) -> None:
        """Load the Gradient Field CNN model with trained weights."""
        # Try different weight file names
        weights_path = None
        for fname in ["gradient_field_cnn_v3_finetuned.pth", "gradient_field_cnn_v2.pth", "weights.pt", "model.pth"]:
            candidate = Path(self.local_path) / fname
            if candidate.exists():
                weights_path = candidate
                break
        
        preprocess_path = Path(self.local_path) / "preprocess.json"
        
        if weights_path is None:
            raise ConfigurationError(
                message=f"No weights file found in {self.local_path}",
                details={"repo_id": self.repo_id}
            )
        
        try:
            # Load preprocessing config
            preprocess_config = {}
            if preprocess_path.exists():
                with open(preprocess_path, "r") as f:
                    preprocess_config = json.load(f)
            
            # Get input size (default 256 for gradient field)
            input_size = preprocess_config.get("input_size", 256)
            if isinstance(input_size, list):
                input_size = input_size[0]
            
            self._resize = transforms.Resize((input_size, input_size))
            
            # Get model parameters from config
            model_params = self.config.get("model_parameters", {})
            depth = model_params.get("depth", 4)
            base_filters = model_params.get("base_filters", 32)
            dropout = model_params.get("dropout", 0.3)
            embedding_dim = model_params.get("embedding_dim", 128)
            
            # Create model
            self._model = CompactGradientNet(
                depth=depth,
                base_filters=base_filters,
                dropout=dropout,
                embedding_dim=embedding_dim
            )
            
            # Load trained weights
            # Note: weights_only=False needed because checkpoint contains numpy types
            state_dict = torch.load(weights_path, map_location=self._device, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(state_dict, dict):
                if "model_state_dict" in state_dict:
                    state_dict = state_dict["model_state_dict"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                elif "model" in state_dict:
                    state_dict = state_dict["model"]
            
            self._model.load_state_dict(state_dict)
            self._model.to(self._device)
            self._model.eval()
            
            # Mark as loaded
            self._predict_fn = self._run_inference
            logger.info(f"Loaded Gradient Field CNN model from {self.repo_id}")
            
        except ConfigurationError:
            raise
        except Exception as e:
            logger.error(f"Failed to load Gradient Field CNN model: {e}")
            raise ConfigurationError(
                message=f"Failed to load model: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
    
    def _rgb_to_luminance(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Convert RGB tensor to luminance using BT.709 coefficients.
        
        Args:
            img_tensor: RGB tensor of shape (3, H, W) with values in [0, 1]
            
        Returns:
            Luminance tensor of shape (1, H, W)
        """
        luminance = (
            self.R_COEFF * img_tensor[0] +
            self.G_COEFF * img_tensor[1] +
            self.B_COEFF * img_tensor[2]
        )
        return luminance.unsqueeze(0)
    
    def _run_inference(
        self,
        luminance_tensor: torch.Tensor,
        explain: bool = False
    ) -> Dict[str, Any]:
        """Run model inference on preprocessed luminance tensor."""
        heatmap = None
        
        if explain:
            # Custom GradCAM implementation for single-logit binary model
            # Using absolute CAM values to capture both positive and negative contributions
            # Target the last Conv2d layer (cnn[-5])
            target_layer = self._model.cnn[-5]
            
            activations = None
            gradients = None
            
            def forward_hook(module, input, output):
                nonlocal activations
                activations = output.detach()
            
            def backward_hook(module, grad_input, grad_output):
                nonlocal gradients
                gradients = grad_output[0].detach()
            
            h_fwd = target_layer.register_forward_hook(forward_hook)
            h_bwd = target_layer.register_full_backward_hook(backward_hook)
            
            try:
                # Forward pass with gradients
                input_tensor = luminance_tensor.clone().requires_grad_(True)
                logits, embedding = self._model(input_tensor)
                prob_fake = torch.sigmoid(logits).item()
                pred_int = 1 if prob_fake >= self._threshold else 0
                
                # Backward pass
                self._model.zero_grad()
                logits.backward()
                
                if gradients is not None and activations is not None:
                    # Compute Grad-CAM weights (global average pooled gradients)
                    weights = gradients.mean(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
                    
                    # Weighted combination of activation maps
                    cam = (weights * activations).sum(dim=1, keepdim=True)  # [1, 1, H, W]
                    
                    # Use absolute values instead of ReLU to capture all contributions
                    # This is important for models where negative gradients carry meaning
                    cam = torch.abs(cam)
                    
                    # Normalize to [0, 1]
                    cam = cam - cam.min()
                    cam_max = cam.max()
                    if cam_max > 0:
                        cam = cam / cam_max
                    
                    # Resize to output size (256x256)
                    cam = F.interpolate(
                        cam,
                        size=(256, 256),
                        mode='bilinear',
                        align_corners=False
                    )
                    
                    heatmap = cam.squeeze().cpu().numpy()
                else:
                    logger.warning("GradCAM: gradients or activations not captured")
                    heatmap = np.zeros((256, 256), dtype=np.float32)
                    
            finally:
                h_fwd.remove()
                h_bwd.remove()
        else:
            with torch.no_grad():
                logits, embedding = self._model(luminance_tensor)
                prob_fake = torch.sigmoid(logits).item()
                pred_int = 1 if prob_fake >= self._threshold else 0
        
        result = {
            "logits": logits.detach().cpu().numpy().tolist() if hasattr(logits, 'detach') else logits.cpu().numpy().tolist(),
            "prob_fake": prob_fake,
            "pred_int": pred_int,
            "embedding": embedding.detach().cpu().numpy().tolist() if explain else embedding.cpu().numpy().tolist()
        }
        
        if heatmap is not None:
            result["heatmap"] = heatmap
        
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
        if self._model is None or self._resize is None:
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
            
            # Resize
            image = self._resize(image)
            
            # Convert to tensor
            img_tensor = transforms.functional.to_tensor(image)
            
            # Convert to luminance
            luminance = self._rgb_to_luminance(img_tensor)
            luminance = luminance.unsqueeze(0).to(self._device)  # Add batch dim
            
            # Run inference
            result = self._run_inference(luminance, explain=explain)
            
            # Standardize output
            labels = self.config.get("labels", {"0": "real", "1": "fake"})
            pred_int = result["pred_int"]
            
            output = {
                "pred_int": pred_int,
                "pred": labels.get(str(pred_int), "unknown"),
                "prob_fake": result["prob_fake"],
                "meta": {
                    "model": self.name,
                    "threshold": self._threshold
                }
            }
            
            # Add heatmap if requested
            if explain and "heatmap" in result:
                heatmap = result["heatmap"]
                output["heatmap_base64"] = heatmap_to_base64(heatmap)
                output["explainability_type"] = "grad_cam"
                output["focus_summary"] = compute_focus_summary(heatmap) + " (edge-based analysis)"
            
            return output
            
        except InferenceError:
            raise
        except Exception as e:
            logger.error(f"Prediction failed for {self.repo_id}: {e}")
            raise InferenceError(
                message=f"Prediction failed: {e}",
                details={"repo_id": self.repo_id, "error": str(e)}
            )
