"""
Model registry for managing loaded models.
"""

import asyncio
import copy
import json
import time
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
from app.services.model_manifest import (
    SubmodelSpec,
    apply_manifest_overrides,
    compute_manifest_hash,
    load_local_manifest,
    parse_submodel_entries,
    resolve_fusion_revision,
)

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
        # Per-repo serving metadata: requested and resolved revision.
        self._serving: Dict[str, Dict[str, Any]] = {}
        self._manifest_hash: Optional[str] = None
        self._load_lock = asyncio.Lock()
        self._hf_service = get_hf_hub_service()
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded
    
    async def load_from_fusion_repo(
        self,
        fusion_repo_id: str,
        force_reload: bool = False,
        fusion_revision: Optional[str] = None,
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
            fusion_revision: Fusion repo revision. Defaults to
                settings.HF_FUSION_REVISION, then the local model manifest,
                then "main".
        """
        total_t0 = time.perf_counter()

        async with self._load_lock:
            if self._is_loaded and not force_reload:
                logger.info("Models already loaded, skipping")
                return
            
            manifest = load_local_manifest(settings.MODEL_MANIFEST_PATH)
            fusion_revision = resolve_fusion_revision(
                fusion_repo_id,
                fusion_revision or settings.HF_FUSION_REVISION,
                manifest,
            )
            logger.info(
                f"Loading models from fusion repo: {fusion_repo_id} "
                f"(revision={fusion_revision or 'main'})"
            )

            # Download fusion repo
            fusion_download_t0 = time.perf_counter()
            fusion_path = await asyncio.to_thread(
                self._hf_service.download_repo, fusion_repo_id, fusion_revision
            )
            logger.info(
                "phase3 fusion_repo_download_duration_seconds=%.3f repo_id=%s",
                time.perf_counter() - fusion_download_t0,
                fusion_repo_id,
            )
            
            # Read fusion config
            fusion_config = self._read_config(fusion_path)
            logger.info(f"Fusion config: {fusion_config}")
            
            # Get submodel specs from config (strings or objects), with
            # revisions pinned by the local manifest taking precedence.
            submodel_specs = apply_manifest_overrides(
                parse_submodel_entries(fusion_config.get("submodels", [])),
                manifest,
            )
            if not submodel_specs:
                raise ConfigurationError(
                    message="Fusion config does not specify any submodels",
                    details={"repo_id": fusion_repo_id}
                )
            
            # Prepare submodels sequentially to avoid concurrent Hugging Face
            # download contention, then load the already-downloaded artifacts in parallel.
            prepare_t0 = time.perf_counter()
            prepared_submodels = []
            for spec in submodel_specs:
                prepared_submodels.append(
                    await self._prepare_submodel(spec)
                )
            logger.info(
                "phase3 submodel_prepare_total_duration_seconds=%.3f count=%d",
                time.perf_counter() - prepare_t0,
                len(prepared_submodels),
            )

            max_concurrent_loads = 2
            semaphore = asyncio.Semaphore(max_concurrent_loads)

            async def load_with_limit(prepared_submodel):
                async with semaphore:
                    return await self._load_prepared_submodel(prepared_submodel)

            load_t0 = time.perf_counter()
            load_results = await asyncio.gather(
                *(load_with_limit(prepared_submodel) for prepared_submodel in prepared_submodels),
                return_exceptions=True
            )
            logger.info(
                "phase3 submodel_parallel_load_total_duration_seconds=%.3f concurrency=%d",
                time.perf_counter() - load_t0,
                max_concurrent_loads,
            )

            errors = [result for result in load_results if isinstance(result, Exception)]
            if errors:
                error_messages = [str(error) for error in errors]
                raise RuntimeError(
                    f"Failed to load one or more submodels: {error_messages}"
                )

            loaded_submodels = {
                wrapper.name: wrapper
                for wrapper in load_results
                if not isinstance(wrapper, Exception)
            }
            
            # Create and load fusion wrapper
            fusion_wrapper_class = get_fusion_wrapper_class(fusion_config)
            logger.info(f"Using fusion wrapper class {fusion_wrapper_class.__name__}")
            fusion_wrapper = fusion_wrapper_class(
                repo_id=fusion_repo_id,
                config=fusion_config,
                local_path=fusion_path
            )
            fusion_wrapper_t0 = time.perf_counter()
            fusion_wrapper.load()
            logger.info(
                "phase3 fusion_wrapper_load_duration_seconds=%.3f",
                time.perf_counter() - fusion_wrapper_t0,
            )
            self._fusion = fusion_wrapper
            self._submodels = loaded_submodels

            serving = {
                fusion_repo_id: {
                    "requested_revision": fusion_revision,
                    "revision": self._hf_service.resolve_commit_sha(fusion_path, fusion_revision),
                }
            }
            for prepared in prepared_submodels:
                serving[prepared["repo_id"]] = {
                    "requested_revision": prepared["revision"],
                    "revision": self._hf_service.resolve_commit_sha(
                        prepared["local_path"], prepared["revision"]
                    ),
                }
            self._serving = serving
            self._manifest_hash = compute_manifest_hash(
                [self._serving_info(fusion_wrapper)]
                + [self._serving_info(w) for w in self._ordered_submodels(prepared_submodels)]
            )
            logger.info(f"Serving manifest_hash={self._manifest_hash}")

            self._is_loaded = True
            logger.info(
                "phase3 load_from_fusion_repo_total_duration_seconds=%.3f",
                time.perf_counter() - total_t0,
            )
            logger.info(f"Successfully loaded {len(self._submodels)} submodels and fusion model")
    
    async def _prepare_submodel(self, spec: SubmodelSpec) -> Dict[str, Any]:
        """
        Download a submodel repository and prepare metadata for loading.

        This stays sequential to avoid concurrent Hugging Face download issues.
        """
        repo_id = spec.repo_id
        logger.info(f"Preparing submodel: {repo_id} (revision={spec.revision or 'main'})")
        prepare_t0 = time.perf_counter()

        local_path = await asyncio.to_thread(
            self._hf_service.download_repo, repo_id, spec.revision
        )
        config = self._read_config(local_path)
        wrapper_class = get_wrapper_class(config)

        # Fusion-manifest overrides reach the wrapper through its config.
        if spec.weights_file or spec.input_size:
            config = copy.deepcopy(config)
            if spec.weights_file:
                config["weights_file"] = spec.weights_file
            if spec.input_size:
                input_cfg = config.get("input")
                config["input"] = dict(input_cfg) if isinstance(input_cfg, dict) else {}
                config["input"]["size"] = spec.input_size

        logger.info(
            "phase3 prepare_submodel_duration_seconds=%.3f repo_id=%s",
            time.perf_counter() - prepare_t0,
            repo_id,
        )

        return {
            "repo_id": repo_id,
            "revision": spec.revision,
            "spec": spec,
            "local_path": local_path,
            "config": config,
            "wrapper_class": wrapper_class,
        }

    async def _load_prepared_submodel(self, prepared_submodel: Dict[str, Any]) -> BaseSubmodelWrapper:
        """
        Load a submodel that has already been downloaded and prepared.
        """
        repo_id = prepared_submodel["repo_id"]
        local_path = prepared_submodel["local_path"]
        config = prepared_submodel["config"]
        wrapper_class = prepared_submodel["wrapper_class"]

        logger.info(f"Loading submodel: {repo_id}")
        logger.info(f"Using wrapper class {wrapper_class.__name__} for {repo_id}")
        load_t0 = time.perf_counter()

        # Create and load wrapper
        wrapper = wrapper_class(
            repo_id=repo_id,
            config=config,
            local_path=local_path
        )
        await asyncio.to_thread(wrapper.load)
        logger.info(f"Loaded submodel: {wrapper.name}")

        spec: Optional[SubmodelSpec] = prepared_submodel.get("spec")
        if spec is not None:
            if spec.weights_file and wrapper.weights_file not in (None, spec.weights_file):
                logger.warning(
                    f"{repo_id}: manifest asks for weights {spec.weights_file} "
                    f"but wrapper loaded {wrapper.weights_file}"
                )
            if spec.input_size and wrapper.input_size not in (None, spec.input_size):
                logger.warning(
                    f"{repo_id}: manifest asks for input_size {spec.input_size} "
                    f"but wrapper uses {wrapper.input_size}"
                )
        logger.info(
            "phase3 load_prepared_submodel_duration_seconds=%.3f repo_id=%s",
            time.perf_counter() - load_t0,
            repo_id,
        )
        return wrapper
    
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
    
    def _ordered_submodels(self, prepared_submodels: List[Dict[str, Any]]) -> List[BaseSubmodelWrapper]:
        """Loaded submodel wrappers in fusion config order."""
        by_repo = {w.repo_id: w for w in self._submodels.values()}
        return [by_repo[p["repo_id"]] for p in prepared_submodels if p["repo_id"] in by_repo]

    def _serving_info(self, wrapper: Any) -> Dict[str, Any]:
        """Repo, revision and artifact details of a loaded wrapper."""
        serving = self._serving.get(wrapper.repo_id, {})
        return {
            "repo_id": wrapper.repo_id,
            "revision": serving.get("revision") or serving.get("requested_revision"),
            "requested_revision": serving.get("requested_revision"),
            "weights_file": getattr(wrapper, "weights_file", None),
            "input_size": getattr(wrapper, "input_size", None),
        }

    def get_manifest_hash(self) -> Optional[str]:
        """Hash of the served repos, revisions and artifacts; None until loaded."""
        return self._manifest_hash if self._is_loaded else None

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
                **self._serving_info(self._fusion),
                "name": self._fusion.name,
                "model_type": "fusion",
                "config": self._fusion.config
            })
        
        # Add submodels
        for name, wrapper in self._submodels.items():
            models.append({
                **self._serving_info(wrapper),
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
