"""Build-time model prefetch utility for reducing cold-start downloads."""

import asyncio

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.services.model_registry import get_model_registry


setup_logging()
logger = get_logger(__name__)


async def main() -> None:
    """Download fusion and submodel repositories into the configured HF cache."""
    logger.info("Starting build-time model prefetch for %s", settings.HF_FUSION_REPO_ID)
    registry = get_model_registry()
    await registry.load_from_fusion_repo(settings.HF_FUSION_REPO_ID, force_reload=True)
    logger.info("Build-time model prefetch completed")


if __name__ == "__main__":
    asyncio.run(main())
