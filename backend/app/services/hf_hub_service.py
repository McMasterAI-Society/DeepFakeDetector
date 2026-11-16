"""
Hugging Face Hub service for downloading model repositories.
"""

import os
import re
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

from app.core.config import settings
from app.core.errors import HuggingFaceDownloadError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Disable symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

_FULL_SHA_RE = re.compile(r"[0-9a-fA-F]{40}")


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

        # Keep Hugging Face process-wide cache settings aligned with app config.
        os.environ.setdefault("HF_HOME", settings.HF_HOME)
        os.environ.setdefault("HF_HUB_CACHE", self.cache_dir)
        
        # Ensure cache directory exists
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"HF Hub service initialized with cache dir: {self.cache_dir}")
    
    def local_dir_for(self, repo_id: str, revision: Optional[str] = None) -> Path:
        """
        Local directory used for a repo at a given revision.

        Unpinned downloads keep the legacy ``<owner>--<name>`` directory so
        existing warm caches stay valid. Pinned downloads use
        ``<owner>--<name>@<revision>``, so changing the revision forces a fresh
        download instead of reusing whatever was cached first.
        """
        repo_name = repo_id.replace("/", "--")
        if revision:
            repo_name = f"{repo_name}@{revision.replace('/', '--')}"
        return Path(self.cache_dir) / repo_name

    def download_repo(
        self,
        repo_id: str,
        revision: Optional[str] = None,
        force_download: bool = False
    ) -> str:
        """
        Download a repository from Hugging Face Hub.
        
        Returns the cached copy for this repo and revision when one exists,
        otherwise downloads it with snapshot_download.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "DeepFakeDetector/test-random-a")
            revision: Git revision (branch, tag, or commit hash). Defaults to "main".
                      Use a commit sha: a pinned branch name is cached on first
                      download and not refreshed.
            force_download: If True, re-download even if cached
            
        Returns:
            Local path to the downloaded repository
            
        Raises:
            HuggingFaceDownloadError: If download fails
        """
        logger.info(f"Downloading repo: {repo_id} (revision={revision}, force={force_download})")
        
        try:
            cached_path = self.get_cached_path(repo_id, revision)
            if cached_path and not force_download:
                logger.info(f"Using cached repo for {repo_id}@{revision or 'main'}: {cached_path}")
                return cached_path

            # Use local_dir instead of cache_dir to avoid symlink issues on Windows
            local_dir = self.local_dir_for(repo_id, revision)
            
            local_path = snapshot_download(
                repo_id=repo_id,
                revision=revision or "main",
                local_dir=str(local_dir),
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
                details={"repo_id": repo_id, "revision": revision, "error": str(e)}
            )
        except Exception as e:
            logger.error(f"Error downloading {repo_id}: {e}")
            raise HuggingFaceDownloadError(
                message=f"Failed to download repository: {repo_id}",
                details={"repo_id": repo_id, "revision": revision, "error": str(e)}
            )
    
    def get_cached_path(self, repo_id: str, revision: Optional[str] = None) -> Optional[str]:
        """
        Get the cached path for a repository at a revision if it exists.
        
        Args:
            repo_id: Hugging Face repository ID
            revision: Git revision; None means the unpinned (legacy) cache dir
            
        Returns:
            Local path if cached, None otherwise
        """
        local_dir = self.local_dir_for(repo_id, revision)
        
        if local_dir.exists() and any(local_dir.iterdir()):
            return str(local_dir)
        return None
    
    def is_cached(self, repo_id: str, revision: Optional[str] = None) -> bool:
        """
        Check if a repository is already cached.
        
        Args:
            repo_id: Hugging Face repository ID
            revision: Git revision; None means the unpinned (legacy) cache dir
            
        Returns:
            True if cached, False otherwise
        """
        return self.get_cached_path(repo_id, revision) is not None

    @staticmethod
    def resolve_commit_sha(local_path: str, revision: Optional[str] = None) -> Optional[str]:
        """
        Best-effort commit sha for a downloaded repo.

        A full 40-character hex revision is returned as is. Otherwise the sha is
        read from the download metadata huggingface_hub writes next to files in
        a ``local_dir`` download. Returns None when it cannot be determined.
        """
        if revision and _FULL_SHA_RE.fullmatch(revision):
            return revision.lower()

        metadata_dir = Path(local_path) / ".cache" / "huggingface" / "download"
        if not metadata_dir.is_dir():
            return None
        candidates = [metadata_dir / "config.json.metadata"]
        candidates += sorted(metadata_dir.glob("*.metadata"))
        for candidate in candidates:
            try:
                first_line = candidate.read_text(encoding="utf-8").splitlines()[0].strip()
            except (OSError, IndexError):
                continue
            if _FULL_SHA_RE.fullmatch(first_line):
                return first_line.lower()
        return None


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
