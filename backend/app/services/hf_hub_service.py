"""
Hugging Face Hub service for downloading model repositories.
"""

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from app.core.config import settings
from app.core.errors import HuggingFaceDownloadError
from app.core.logging import get_logger

logger = get_logger(__name__)


class HFHubService:
    """
    Service for interacting with Hugging Face Hub.
    
    Handles downloading model repositories and caching them locally.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, token: Optional[str] = None):
        """
        Initialize the HF Hub service.
        
        Args:
            cache_dir: Local directory for caching downloads. 
                      Defaults to settings.HF_CACHE_DIR
            token: Hugging Face API token for private repos.
                  Defaults to settings.HF_TOKEN
        """
        self.cache_dir = cache_dir or settings.HF_CACHE_DIR
        self.token = token or settings.HF_TOKEN
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"HF Hub service initialized with cache dir: {self.cache_dir}")
    
    def download_repo(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        force_download: bool = False
    ) -> str:
        """
        Download a repository from Hugging Face Hub.
        
        Uses snapshot_download which handles caching automatically.
        If the repo is already cached and not stale, it returns the cached path.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "DeepFakeDetector/test-random-a")
            revision: Git revision (branch, tag, or commit hash). Defaults to "main"
            force_download: If True, re-download even if cached
            
        Returns:
            Local path to the downloaded repository
            
        Raises:
            HuggingFaceDownloadError: If download fails
        """
        logger.info(f"Downloading repo: {repo_id} (revision={revision}, force={force_download})")
        
        try:
            local_path = snapshot_download(
                repo_id=repo_id,
                revision=revision or "main",
                cache_dir=self.cache_dir,
                token=self.token,
                force_download=force_download,
                local_files_only=False
            )
            
            logger.info(f"Downloaded {repo_id} to {local_path}")
            return local_path
            
        except HfHubHTTPError as e:
            logger.error(f"HTTP error downloading {repo_id}: {e}")
            raise HuggingFaceDownloadError(
                message=f"Failed to download repository: {repo_id}",
                details={"repo_id": repo_id, "error": str(e)}
            )
        except Exception as e:
            logger.error(f"Error downloading {repo_id}: {e}")
            raise HuggingFaceDownloadError(
                message=f"Failed to download repository: {repo_id}",
                details={"repo_id": repo_id, "error": str(e)}
            )
    
    def get_cached_path(self, repo_id: str) -> Optional[str]:
        """
        Get the cached path for a repository if it exists.
        
        Args:
            repo_id: Hugging Face repository ID
            
        Returns:
            Local path if cached, None otherwise
        """
        try:
            # Try to get from cache without downloading
            local_path = snapshot_download(
                repo_id=repo_id,
                cache_dir=self.cache_dir,
                local_files_only=True
            )
            return local_path
        except Exception:
            return None
    
    def is_cached(self, repo_id: str) -> bool:
        """
        Check if a repository is already cached.
        
        Args:
            repo_id: Hugging Face repository ID
            
        Returns:
            True if cached, False otherwise
        """
        return self.get_cached_path(repo_id) is not None


# Global singleton instance
_hf_hub_service: Optional[HFHubService] = None


def get_hf_hub_service() -> HFHubService:
    """
    Get the global HF Hub service instance.
    
    Returns:
        HFHubService instance
    """
    global _hf_hub_service
    if _hf_hub_service is None:
        _hf_hub_service = HFHubService()
    return _hf_hub_service
