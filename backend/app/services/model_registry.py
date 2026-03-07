"""
Model registry for managing loaded models.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from app.core.config import settings
from app.core.errors import ModelNotFoundError, ModelNotLoadedError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseSubmodelWrapper, BaseFusionWrapper
from app.models.wrappers.dummy_random_wrapper import DummyRandomWrapper
from app.models.wrappers.dummy_majority_fusion_wrapper import DummyMajorityFusionWrapper
from app.models.wrappers.logreg_fusion_wrapper import LogRegFusionWrapper
# Real production wrappers
from app.models.wrappers.cnn_transfer_wrapper import CNNTransferWrapper
from app.models.wrappers.deit_distilled_wrapper import DeiTDistilledWrapper
from app.models.wrappers.vit_base_wrapper import ViTBaseWrapper
from app.models.wrappers.gradfield_cnn_wrapper import GradfieldCNNWrapper
from app.services.hf_hub_service import get_hf_hub_service

logger = get_logger(__name__)


def get_wrapper_class(config: Dict[str, Any]) -> Type[BaseSubmodelWrapper]:
    """
    Select the appropriate wrapper class based on model config.
    
    Uses architecture hints or model_type to dispatch to the correct wrapper.
    Falls back to DummyRandomWrapper if no match found (useful for testing).
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Wrapper class (not instance)
    """
    # Check various config fields that might indicate model type
    arch = config.get("arch", "").lower()
    model_type = config.get("type", "").lower()
    model_class = config.get("model_class", "").lower()
    model_name = config.get("model_name", "").lower()
    library = config.get("library", "").lower()
    
    # EfficientNet / CNN Transfer
    if "efficientnet" in arch or "cnn-transfer" in model_type or "efficientnet" in model_name:
        return CNNTransferWrapper
    
    # DeiT Distilled
    if "deit" in arch or "deit-distilled" in model_type or "deit" in model_name:
        return DeiTDistilledWrapper
    
    # ViT Base (check vit but not deit)
    if (("vit" in arch or "vit" in model_name) and "deit" not in arch and "deit" not in model_name) or "vit-base" in model_type:
        return ViTBaseWrapper
    
    # Gradient Field CNN
    if "gradient" in arch or "gradientnet" in model_class or "gradfield" in model_type or "gradient" in model_name:
        return GradfieldCNNWrapper
    
    # Fallback to dummy wrapper
    logger.warning(f"No matching wrapper for config, using DummyRandomWrapper: {config}")
    return DummyRandomWrapper


def get_fusion_wrapper_class(config: Dict[str, Any]) -> Type[BaseFusionWrapper]:
    """
    Select the appropriate fusion wrapper class based on config.
    
    Args:
        config: Fusion model configuration dictionary
        
    Returns:
        Fusion wrapper class (not instance)
    """
    fusion_type = config.get("type", "").lower()
    
    # Logistic regression stacking fusion
    if "probability_stacking" in fusion_type or "logreg" in fusion_type:
        return LogRegFusionWrapper
    
    # Majority vote fusion
    if "majority" in fusion_type:
        return DummyMajorityFusionWrapper
    
    # Default to majority fusion
    logger.warning(f"Unknown fusion type, using DummyMajorityFusionWrapper: {fusion_type}")
    return DummyMajorityFusionWrapper


class ModelRegistry:
    """
    Central registry for all loaded models.
    
    Manages downloading, loading, and accessing models from Hugging Face Hub.
    This is the single source of truth for model state.
    """
    
    def __init__(self):
        self._fusion: Optional[BaseFusionWrapper] = None
        self._submodels: Dict[str, BaseSubmodelWrapper] = {}
        self._is_loaded: bool = False
        self._load_lock = asyncio.Lock()
        self._hf_service = get_hf_hub_service()
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded
    
    async def load_from_fusion_repo(
        self,
        fusion_repo_id: str,
        force_reload: bool = False
    ) -> None:
        """
        Load fusion model and all submodels from a fusion repository.
        
        This is the main entry point for loading models. It:
        1. Downloads the fusion repo and reads its config.json
        2. Extracts submodel repo IDs from config
        3. Downloads and loads each submodel
        4. Loads the fusion model
        
        Args:
            fusion_repo_id: Hugging Face repository ID for fusion model
            force_reload: If True, reload even if already loaded
        """
        async with self._load_lock:
            if self._is_loaded and not force_reload:
                logger.info("Models already loaded, skipping")
                return
            
            logger.info(f"Loading models from fusion repo: {fusion_repo_id}")
            
            # Download fusion repo
            fusion_path = await asyncio.to_thread(
                self._hf_service.download_repo, fusion_repo_id
            )
            
            # Read fusion config
            fusion_config = self._read_config(fusion_path)
            logger.info(f"Fusion config: {fusion_config}")
            
            # Get submodel repo IDs from config
            submodel_repos = fusion_config.get("submodels", [])
            if not submodel_repos:
                raise ConfigurationError(
                    message="Fusion config does not specify any submodels",
                    details={"repo_id": fusion_repo_id}
                )
            
            # Download and load each submodel
            for submodel_repo_id in submodel_repos:
                await self._load_submodel(submodel_repo_id)
            
            # Create and load fusion wrapper
            fusion_wrapper_class = get_fusion_wrapper_class(fusion_config)
            logger.info(f"Using fusion wrapper class {fusion_wrapper_class.__name__}")
            self._fusion = fusion_wrapper_class(
                repo_id=fusion_repo_id,
                config=fusion_config,
                local_path=fusion_path
            )
            self._fusion.load()
            
            self._is_loaded = True
            logger.info(f"Successfully loaded {len(self._submodels)} submodels and fusion model")
    
    async def _load_submodel(self, repo_id: str) -> None:
        """
        Download and load a single submodel.
        
        Uses the config to determine the correct wrapper class.
        
        Args:
            repo_id: Hugging Face repository ID for the submodel
        """
        logger.info(f"Loading submodel: {repo_id}")
        
        # Download the repo
        local_path = await asyncio.to_thread(
            self._hf_service.download_repo, repo_id
        )
        
        # Read config
        config = self._read_config(local_path)
        
        # Select appropriate wrapper class based on config
        wrapper_class = get_wrapper_class(config)
        logger.info(f"Using wrapper class {wrapper_class.__name__} for {repo_id}")
        
        # Create and load wrapper
        wrapper = wrapper_class(
            repo_id=repo_id,
            config=config,
            local_path=local_path
        )
        wrapper.load()
        
        # Store by short name
        self._submodels[wrapper.name] = wrapper
        logger.info(f"Loaded submodel: {wrapper.name}")
    
    def _read_config(self, local_path: str) -> Dict[str, Any]:
        """
        Read config.json from a local model path.
        
        Args:
            local_path: Path to the downloaded model
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(local_path) / "config.json"
        
        if not config_path.exists():
            logger.warning(f"config.json not found at {config_path}, using empty config")
            return {}
        
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all loaded models.
        
        Returns:
            List of model info dictionaries
        """
        models = []
        
        # Add fusion model
        if self._fusion:
            models.append({
                "repo_id": self._fusion.repo_id,
                "name": self._fusion.name,
                "model_type": "fusion",
                "config": self._fusion.config
            })
        
        # Add submodels
        for name, wrapper in self._submodels.items():
            models.append({
                "repo_id": wrapper.repo_id,
                "name": name,
                "model_type": "submodel",
                "config": wrapper.config
            })
        
        return models
    
    def get_submodel(self, key: str) -> BaseSubmodelWrapper:
        """
        Get a submodel by name or repo_id.
        
        Args:
            key: Submodel name or full repo_id
            
        Returns:
            Submodel wrapper
            
        Raises:
            ModelNotFoundError: If submodel not found
            ModelNotLoadedError: If models not loaded
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(
                message="Models not loaded yet",
                details={"requested_model": key}
            )
        
        # Try by name first
        if key in self._submodels:
            return self._submodels[key]
        
        # Try by repo_id
        for name, wrapper in self._submodels.items():
            if wrapper.repo_id == key:
                return wrapper
        
        raise ModelNotFoundError(
            message=f"Submodel not found: {key}",
            details={
                "requested_model": key,
                "available_models": list(self._submodels.keys())
            }
        )
    
    def get_all_submodels(self) -> Dict[str, BaseSubmodelWrapper]:
        """
        Get all loaded submodels.
        
        Returns:
            Dictionary mapping name to submodel wrapper
            
        Raises:
            ModelNotLoadedError: If models not loaded
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(message="Models not loaded yet")
        return self._submodels.copy()
    
    def get_fusion(self) -> BaseFusionWrapper:
        """
        Get the fusion model.
        
        Returns:
            Fusion model wrapper
            
        Raises:
            ModelNotLoadedError: If models not loaded
        """
        if not self._is_loaded or self._fusion is None:
            raise ModelNotLoadedError(message="Fusion model not loaded yet")
        return self._fusion
    
    def get_submodel_names(self) -> List[str]:
        """Get list of loaded submodel names."""
        return list(self._submodels.keys())
    
    def get_fusion_repo_id(self) -> Optional[str]:
        """Get the fusion repo ID if loaded."""
        return self._fusion.repo_id if self._fusion else None


# Global singleton instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """
    Get the global model registry instance.
    
    Returns:
        ModelRegistry instance
    """
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
