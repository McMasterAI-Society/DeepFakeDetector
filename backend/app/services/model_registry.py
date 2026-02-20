"""
Model registry for managing loaded models.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.errors import ModelNotFoundError, ModelNotLoadedError, ConfigurationError
from app.core.logging import get_logger
from app.models.wrappers.base_wrapper import BaseSubmodelWrapper, BaseFusionWrapper
from app.models.wrappers.dummy_random_wrapper import DummyRandomWrapper
from app.models.wrappers.dummy_majority_fusion_wrapper import DummyMajorityFusionWrapper
from app.services.hf_hub_service import get_hf_hub_service

logger = get_logger(__name__)


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
            self._fusion = DummyMajorityFusionWrapper(
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
        
        # Create and load wrapper
        wrapper = DummyRandomWrapper(
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
