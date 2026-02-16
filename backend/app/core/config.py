"""
Application configuration with environment variable support.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Hugging Face configuration
    HF_FUSION_REPO_ID: str = "DeepFakeDetector/fusion-majority-test"
    HF_CACHE_DIR: str = ".hf_cache"
    HF_TOKEN: Optional[str] = None
    
    # Application configuration
    ENABLE_DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
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
