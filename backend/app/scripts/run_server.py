"""Runtime launcher with extra startup timing for container boot analysis."""

import os
import time
from datetime import datetime, timezone

from app.core.config import settings


START_T0 = time.perf_counter()


def _log(message: str) -> None:
    print(
        f"{datetime.now(timezone.utc).isoformat()} | INFO     | app.scripts.run_server | {message}",
        flush=True,
    )


def main() -> None:
    port = os.environ.get("PORT", str(settings.PORT))
    _log(f"launcher_start port={port} host={settings.HOST}")

    import uvicorn

    _log(f"uvicorn_import_complete duration_seconds={time.perf_counter() - START_T0:.3f}")
    _log(
        f"calling_uvicorn_run app=app.main:app host={settings.HOST} port={port} reload={settings.ENABLE_DEBUG}"
    )
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=int(port),
        reload=settings.ENABLE_DEBUG,
        log_level="info",
    )


if __name__ == "__main__":
    main()