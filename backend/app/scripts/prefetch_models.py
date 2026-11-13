"""Build-time model prefetch utility for reducing cold-start downloads."""

import asyncio

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.services.model_registry import get_model_registry


setup_logging()
logger = get_logger(__name__)


async def main() -> None:
    """Download the production model (MODEL_MODE) into the configured HF cache."""
    registry = get_model_registry()
    if settings.MODEL_MODE == "foundation-probe":
        logger.info("Starting build-time model prefetch for %s", settings.HF_PROBE_REPO_ID)
        await registry.load_foundation_probe(force_reload=True)
    else:
        logger.info("Starting build-time model prefetch for %s", settings.HF_FUSION_REPO_ID)
        await registry.load_from_fusion_repo(settings.HF_FUSION_REPO_ID, force_reload=True)
    logger.info("Build-time model prefetch completed")


if __name__ == "__main__":
    asyncio.run(main())
