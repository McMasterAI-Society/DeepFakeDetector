#!/usr/bin/env python3
"""
Fetch production submodels from Hugging Face Hub.

This script downloads the trained submodels to the local .hf_cache directory
for use in the DeepFake Detector backend.

Repos:
- DeepFakeDetector/deit-distilled
- DeepFakeDetector/cnn-transfer
- DeepFakeDetector/gradfield-cnn
- DeepFakeDetector/vit-base
- DeepFakeDetector/fusion-majority-test (TEMP - update to real fusion model later)

Usage:
    python scripts/fetch_production_models.py
    python scripts/fetch_production_models.py --force  # Re-download even if cached
    python scripts/fetch_production_models.py --cache-dir /custom/path
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download, list_repo_files
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
    sys.exit(1)


# Production submodel repositories
PRODUCTION_SUBMODELS = [
    "DeepFakeDetector/deit-distilled",
    "DeepFakeDetector/cnn-transfer",
    "DeepFakeDetector/gradfield-cnn",
    "DeepFakeDetector/vit-base",
]

# Fusion model repository
# TODO: Update this to the actual trained fusion model once available
#       e.g., DeepFakeDetector/fusion-metamodel or DeepFakeDetector/fusion-logreg
FUSION_MODEL = "DeepFakeDetector/fusion-majority-test"

# Default cache directory (relative to repo root)
DEFAULT_CACHE_DIR = Path(__file__).parent.parent / "backend" / ".hf_cache"


def download_repo(repo_id: str, cache_dir: Path, force: bool = False) -> str:
    """
    Download a single repository from Hugging Face Hub.
    
    Args:
        repo_id: HuggingFace repository ID (e.g., "DeepFakeDetector/vit-base")
        cache_dir: Local directory for caching
        force: If True, re-download even if cached
        
    Returns:
        Local path to downloaded repository
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {repo_id}")
    print(f"{'='*60}")
    
    # First, list files in the repo to show what we're getting
    try:
        files = list_repo_files(repo_id)
        print(f"Files in repo:")
        for f in files:
            print(f"  - {f}")
    except Exception as e:
        print(f"  Could not list files: {e}")
    
    # Download the repo
    # Use local_dir instead of cache_dir to avoid symlink issues on Windows
    import os
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    try:
        # Create a direct download path to avoid symlink issues
        repo_name = repo_id.replace("/", "--")
        local_dir = cache_dir / repo_name
        
        local_path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            force_download=force,
            local_files_only=False,
        )
        print(f"\n✓ Downloaded to: {local_path}")
        
        # Show downloaded files
        local_path_obj = Path(local_path)
        print(f"\nLocal files:")
        for f in local_path_obj.rglob("*"):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.relative_to(local_path_obj)} ({size_kb:.1f} KB)")
        
        return local_path
        
    except HfHubHTTPError as e:
        print(f"\n✗ HTTP Error: {e}")
        if "404" in str(e):
            print(f"  Repository '{repo_id}' not found. Check if it exists and is public.")
        raise
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download production submodels from Hugging Face Hub"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache directory (default: {DEFAULT_CACHE_DIR})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached"
    )
    parser.add_argument(
        "--repo",
        type=str,
        help="Download only a specific repo (e.g., DeepFakeDetector/vit-base)"
    )
    
    args = parser.parse_args()
    
    # Ensure cache directory exists
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DeepFake Detector - Production Model Fetcher")
    print("="*60)
    print(f"Cache directory: {args.cache_dir.resolve()}")
    print(f"Force download: {args.force}")
    
    # Determine which repos to download
    repos = [args.repo] if args.repo else PRODUCTION_SUBMODELS + [FUSION_MODEL]
    
    print(f"\nRepos to download ({len(repos)}):")
    for repo in repos:
        print(f"  - {repo}")
    
    # Download each repo
    results = {}
    for repo_id in repos:
        try:
            local_path = download_repo(repo_id, args.cache_dir, args.force)
            results[repo_id] = {"status": "success", "path": local_path}
        except Exception as e:
            results[repo_id] = {"status": "failed", "error": str(e)}
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    success = 0
    failed = 0
    for repo_id, result in results.items():
        if result["status"] == "success":
            print(f"✓ {repo_id}")
            print(f"    → {result['path']}")
            success += 1
        else:
            print(f"✗ {repo_id}")
            print(f"    Error: {result['error']}")
            failed += 1
    
    print(f"\nTotal: {success} successful, {failed} failed")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
