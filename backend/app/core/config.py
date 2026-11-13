"""
Application configuration with environment variable support.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Literal, Optional


def _default_hf_cache_dir() -> str:
    """Prefer persistent storage on HF Spaces when available."""
    if os.path.isdir("/data"):
        return "/data/.hf_cache"
    return "/app/.hf_cache"


def _default_hf_home() -> str:
    """Prefer persistent Hugging Face home on HF Spaces when available."""
    if os.path.isdir("/data"):
        return "/data/.cache/huggingface"
    return "/app/.hf_cache/hf_home"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Hugging Face configuration
    # Available fusion models:
    #   - DeepFakeDetector/fusion-logreg-final (Logistic Regression - default)
    #   - DeepFakeDetector/fusion-meta-final (Meta-classifier)
    HF_FUSION_REPO_ID: str = "DeepFakeDetector/fusion-logreg-final"
    HF_CACHE_DIR: str = _default_hf_cache_dir()
    HF_HOME: str = _default_hf_home()
    HF_PERSISTENT_CACHE: bool = True
    HF_PERSISTENT_CACHE_ROOT: str = "/data"
    HF_TOKEN: Optional[str] = None
    # Optional git revision (ideally a commit sha) of the fusion repo. When unset,
    # the revision from model-manifest.json is used, falling back to "main".
    HF_FUSION_REVISION: Optional[str] = None
    # Optional path to a model manifest pinning repo revisions. Defaults to
    # backend/model-manifest.json when that file exists.
    MODEL_MANIFEST_PATH: Optional[str] = None
    # Gradient-field CNN weights file override. When unset, the wrapper uses
    # config/manifest "weights_file", then its legacy preference list
    # (gradient_field_cnn_v3_finetuned.pth first). The deployed fusion was fit on
    # gradient_field_cnn_v2.pth at 224px.
    GRADFIELD_WEIGHTS_FILE: Optional[str] = None

    # Which production model to serve:
    #   foundation-probe: frozen CLIP-B/16 + DINOv3-B/16 features with a calibrated
    #                     linear head (HF_PROBE_REPO_ID). Default.
    #   legacy-fusion:    the 4-submodel logistic-regression fusion
    #                     (HF_FUSION_REPO_ID), kept for rollback and A/B tests.
    MODEL_MODE: Literal["foundation-probe", "legacy-fusion"] = "foundation-probe"
    # If the probe fails to load, load the legacy fusion instead of serving nothing.
    MODEL_FALLBACK_TO_LEGACY: bool = True
    # Head repo (config.json + head.npz + licence). Backbones come from their
    # upstream timm repos at the revisions pinned in its config/the manifest.
    HF_PROBE_REPO_ID: str = "DeepFakeDetector/fm-probe-clip-dinov3"
    # Optional head repo revision; else the manifest's "probe" revision, else main.
    HF_PROBE_REVISION: Optional[str] = None
    # Local directory with config.json + head.npz to use instead of the head repo
    # (testing before upload). Nothing is downloaded for the head when set.
    PROBE_LOCAL_PATH: Optional[str] = None
    # Directory searched first for <key>.fp16.safetensors / <key>.fp16.pt backbone
    # weights (e.g. the experiment's weights/); otherwise upstream timm repos.
    PROBE_BACKBONE_DIR: Optional[str] = None
    # "auto" (CUDA fp16 when available, else CPU fp32), "cuda" or "cpu".
    PROBE_DEVICE: str = "auto"
    # Verdict band edges "lo,hi" on prob_fake: real < lo <= inconclusive < hi <= ai.
    # Default: config.json "verdict_bands", else 0.2,0.7.
    PROBE_VERDICT_BANDS: Optional[str] = None
    # Compute probe heatmaps on /predict when explain=true. Off by default: the
    # gradient explanation costs ~2.8x a plain prediction (RTX 4060 Laptop,
    # 0.29 s -> 0.82 s), so heatmaps are served on demand by POST /explain.
    PROBE_EXPLAIN_ON_PREDICT: bool = False

    # Google Gemini API configuration
    GOOGLE_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-2.5-flash"
    
    @property
    def llm_enabled(self) -> bool:
        """Check if LLM explanations are available."""
        return self.GOOGLE_API_KEY is not None and len(self.GOOGLE_API_KEY) > 0
    
    # Application configuration
    ENABLE_DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS configuration
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,https://www.deepfake-detector.app,https://deepfake-detector.app"
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
    
    # API configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "DeepFake Detector API"
    VERSION: str = "0.1.0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
