"""
Application configuration with environment variable support.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


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
    
    # Prediction store / feedback (see app/services/store.py)
    # DATA_DIR: where app.db and consented uploads live. Empty -> /data when
    # writable (HF Spaces persistent volume), else ./.local_data.
    DATA_DIR: Optional[str] = None
    STORE_ENABLED: bool = True
    # Rows, feedback and stored images older than this are purged. 0 disables purge.
    RETENTION_DAYS: int = 90
    PURGE_INTERVAL_HOURS: float = 24.0
    # Longest side (px) of images kept for research; larger images are downscaled.
    STORED_IMAGE_MAX_SIDE: int = 2048
    STORED_IMAGE_WEBP_QUALITY: int = 90
    # Salt for hashing client IPs (raw IPs are never stored). Empty -> random salt kept in the DB.
    CLIENT_HASH_SALT: Optional[str] = None
    # Serving variant recorded with each prediction (prod / candidate / shadow).
    PREDICTION_VARIANT: str = "prod"

    # Admin dashboard. Disabled unless ADMIN_PASSWORD_HASH is set.
    # Generate the hash with: python -m app.scripts.hash_admin_password
    ADMIN_PASSWORD_HASH: Optional[str] = None
    # HMAC key for admin bearer tokens. Empty -> random per process (tokens die on restart).
    ADMIN_TOKEN_SECRET: Optional[str] = None
    ADMIN_TOKEN_TTL_SECONDS: int = 4 * 3600
    ADMIN_LOGIN_MAX_FAILURES: int = 5
    ADMIN_LOGIN_LOCKOUT_SECONDS: int = 900
    # Probability band treated as "uncertain" in admin filters and stats.
    UNCERTAIN_LOW: float = 0.35
    UNCERTAIN_HIGH: float = 0.65

    # Rate limiting (in-process token bucket per client IP). Format "<count>/<period>",
    # period = second|minute|hour|day or a number of seconds.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PREDICT: str = "20/minute"
    RATE_LIMIT_EXPLAIN: str = "20/minute"
    RATE_LIMIT_FEEDBACK: str = "30/minute"
    RATE_LIMIT_ADMIN_LOGIN: str = "10/minute"
    # Comma-separated proxy IPs/CIDRs whose X-Forwarded-For is trusted. "*" trusts the
    # immediate peer (use behind a single reverse proxy, e.g. HF Spaces).
    # Empty -> X-Forwarded-For is ignored and the socket peer address is used.
    TRUSTED_PROXIES: str = ""

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
