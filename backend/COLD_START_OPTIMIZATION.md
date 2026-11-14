# Cold Start Optimization Implementation Guide (HF Spaces GPU)

## Goal

Reduce end-to-end cold start time for the backend on Hugging Face Spaces GPU while preserving inference quality and endpoint behavior.

This guide is focused only on cold start optimization for the current FastAPI architecture.

## Baseline From Current Logs

Source log window:
- Build queued at 2026-04-20 04:23:34
- Application startup begins at 2026-04-20 04:24:02
- Models loaded successfully at 2026-04-20 04:25:36

### Baseline Timing Summary

| Segment | Start | End | Duration | Notes |
|---|---:|---:|---:|---|
| Queue/build to app startup | 04:23:34 | 04:24:02 | 28s | Includes scheduling, build finalization, image start |
| App startup to model-ready | 04:24:02 | 04:25:36 | 94s | Time from uvicorn start message to models loaded |
| API model load phase | 04:25:15 | 04:25:36 | 21s | From "Starting DeepFake Detector API..." to "Models loaded successfully!" |

### Build Stage Durations Visible In Logs

| Build Stage | Duration |
|---|---:|
| Restoring cache | 19.5s |
| COPY source to /app | 0.0s |
| mkdir/chown/chmod step | 0.1s |
| Pushing image | 0.7s |
| Exporting cache | 0.1s |
| Total visible timed stages | 20.4s |

Note:
- Several Docker steps were cache hits and reported as CACHED without explicit timing.
- "Application startup complete" appears immediately after model load logs; no explicit timestamp is printed, so 04:25:36 is used as the practical ready time.

### Model Load Breakdown (Current)

| Model | Start | End | Duration | Observation |
|---|---:|---:|---:|---|
| Fusion repo config | 04:25:15 | 04:25:16 | 1s | Fast |
| cnn-transfer-final | 04:25:16 | 04:25:17 | 1s | Fast |
| vit-base-final | 04:25:17 | 04:25:30 | 13s | Dominant bottleneck |
| deit-distilled-final | 04:25:30 | 04:25:35 | 5s | Moderate |
| gradfield-cnn-final | 04:25:35 | 04:25:35 | <1s | Fast |
| fusion model load | 04:25:35 | 04:25:36 | 1s | Fast |
| Total model load | 04:25:15 | 04:25:36 | 21s | Sequential loading |

## Current Bottlenecks

1. Dominant pre-app startup delay before Python module import begins.
2. Build-time prefetch cost when cache layers miss (extra build wall time).
3. Model loading is no longer dominant (~4s with current cache and bounded parallel load).
4. Cold-start variance likely includes platform scheduling/provisioning overhead.

## Implementation Plan

## Phase 1: Remove Runtime Model Downloads (Highest Impact)

### 1.1 Add model prefetch script

Create file: app/scripts/prefetch_models.py

Purpose:
- Download fusion repo and all submodel repos at build time into HF_CACHE_DIR.
- Ensure cold start does not wait on remote model downloads.

Implementation:

```python
import asyncio
from app.core.config import settings
from app.services.model_registry import get_model_registry


async def main() -> None:
    registry = get_model_registry()
    await registry.load_from_fusion_repo(settings.HF_FUSION_REPO_ID, force_reload=True)


if __name__ == "__main__":
    asyncio.run(main())
```

### 1.2 Update Dockerfile for build-time prefetch

Target file: Dockerfile

Key changes:
1. Keep dependency installation in a stable cache layer.
2. Copy only application code needed for prefetch before full source copy.
3. Run prefetch script during build with HF cache directory set.
4. Keep ownership and permissions for user uid 1000.

Implementation sketch:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    HF_CACHE_DIR=/app/.hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
ENV PATH="/home/user/.local/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy app code required for prefetch
COPY app /app/app
COPY start.sh /app/start.sh

RUN mkdir -p /app/.hf_cache

# Build-time model prefetch (requires public repos or HF token in build env)
RUN python -m app.scripts.prefetch_models

RUN chown -R user:user /app && chmod +x /app/start.sh
USER user

EXPOSE 7860
CMD ["./start.sh"]
```

Notes:
- If private model repos are used, build needs HF_TOKEN.
- This increases image size but reduces startup wait caused by downloads.

### 1.3 Verify HF cache is reused at runtime

Target file: app/services/hf_hub_service.py

Behavior to enforce:
- Keep deterministic local_dir path under /app/.hf_cache.
- Log cache hits clearly before download attempt.

Add logic before snapshot_download call:

```python
cached = self.get_cached_path(repo_id)
if cached and not force_download:
    logger.info(f"Using cached repo for {repo_id}: {cached}")
    return cached
```

## Phase 2: Parallelize Submodel Loading

Target file: app/services/model_registry.py

Current behavior:
- Submodels are loaded one by one.

New behavior:
- Load submodels concurrently with bounded parallelism.

Implementation steps:
1. Add a semaphore, for example max concurrency 2.
2. Replace sequential loop with asyncio.gather.
3. Keep deterministic final registration and clear error propagation.

Implementation sketch:

```python
sem = asyncio.Semaphore(2)

async def _load_with_limit(repo_id: str) -> None:
    async with sem:
        await self._load_submodel(repo_id)

tasks = [_load_with_limit(repo_id) for repo_id in submodel_repos]
results = await asyncio.gather(*tasks, return_exceptions=True)
errors = [r for r in results if isinstance(r, Exception)]
if errors:
    raise RuntimeError(f"Failed to load one or more submodels: {errors}")
```

Reason for bounded parallelism:
- Reduces startup time without overwhelming memory/network in GPU Space containers.

## Phase 3: Add Startup Instrumentation For Reliable Comparisons

Target file: app/main.py

Add timing markers:
- App startup begin timestamp.
- Model loading start and end.
- Total lifespan startup duration.

Implementation sketch:

```python
import time

startup_t0 = time.perf_counter()
...
model_t0 = time.perf_counter()
await registry.load_from_fusion_repo(settings.HF_FUSION_REPO_ID)
model_dt = time.perf_counter() - model_t0
logger.info(f"Model load duration_seconds={model_dt:.3f}")
...
startup_dt = time.perf_counter() - startup_t0
logger.info(f"Startup total duration_seconds={startup_dt:.3f}")
```

## Phase 4: Use Persistent Storage Cache (/data)

Goal:
- Make /data the primary cache location so model artifacts survive container rebuilds/restarts.

Target files:
- app/core/config.py
- start.sh
- app/services/hf_hub_service.py

Plan:
1. Prefer /data-backed cache paths when available:
    - HF_HOME=/data/.cache/huggingface
    - HF_CACHE_DIR=/data/.hf_cache
2. Keep fallback to /app/.hf_cache when /data is unavailable.
3. Ensure startup creates/chowns cache directories safely.
4. Keep cache-hit logging so verification remains explicit in logs.

Expected impact:
- Faster warm boots across deploys.
- Lower risk of repeated network fetch for large model files.

## Phase 5: Decouple Build From Prefetch

Goal:
- Reduce rebuild penalty from model prefetch while keeping runtime fast.

Target file:
- Dockerfile

Plan:
1. Make build-time prefetch optional via ARG/ENV flag.
2. Default to skipping build prefetch when persistent /data cache is enabled.
3. Keep one-time warm path at runtime (guarded by cache/sentinel file in /data).

Expected impact:
- Faster image rebuild/push cycles.
- Better developer iteration speed without sacrificing warmed production startup.

## Phase 6: Platform/GPU Startup Characterization

Goal:
- Quantify how much of remaining cold start is platform provisioning vs app code.

Plan:
1. Run repeated cold starts with identical image on current T4.
2. If available, test one higher-tier GPU Space and compare only phase3 markers.
3. Record variance in:
    - Application Startup -> module_import_start
    - module_import_complete -> startup complete

Notes:
- GPU type can affect model initialization time.
- The measured dominant delay currently occurs before app module import, so platform scheduling/provisioning is likely the bigger lever than model code tuning.

## Phase 7: Runtime Hygiene (Completed)

Completed changes:
1. Set valid OMP default in start.sh.
2. Pin scikit-learn to 1.6.1 for pickle compatibility.

## Validation and Benchmark Protocol

Use the same procedure before and after changes.

1. Force a cold deployment in HF Space.
2. Record these timestamps from logs:
   - Build queued time
   - Application startup time
   - Starting DeepFake Detector API
   - Models loaded successfully
   - Application startup complete
3. Compute:
   - Queue/build to app startup
   - App startup to model-ready
   - API model load phase
4. Capture per-model load durations from logs.
5. Save a comparison table in this file.

## Phase 1 Results From Latest Logs

Source log window:
- Build queued at 2026-04-20 05:04:31
- Application startup begins at 2026-04-20 05:05:07
- Models loaded successfully at 2026-04-20 05:06:46

### Phase 1 Timing Summary

| Segment | Start | End | Duration | Notes |
|---|---:|---:|---:|---|
| Queue/build to app startup | 05:04:31 | 05:05:07 | 36s | Includes scheduling, build finalization, image start |
| App startup to model-ready | 05:05:07 | 05:06:46 | 99s | Time from uvicorn start message to models loaded |
| API model load phase | 05:06:41 | 05:06:46 | 5s | From "Starting DeepFake Detector API..." to "Models loaded successfully!" |

### Phase 1 Observations

- All Hugging Face repos were served from cache at runtime, confirming the build-time prefetch is working.
- The previous runtime download cost was eliminated from startup.
- The remaining startup time is now dominated by model wrapper initialization and import/init overhead rather than repo downloads.

## Phase 2 Results From Latest Logs

Source log window:
- Build queued at 2026-04-20 05:46:19
- Application startup begins at 2026-04-20 05:48:18
- Models loaded successfully at 2026-04-20 05:49:56

### Phase 2 Timing Summary

| Segment | Start | End | Duration | Notes |
|---|---:|---:|---:|---|
| Queue/build to app startup | 05:46:19 | 05:48:18 | 119s | Includes scheduling, build finalization, image start |
| App startup to model-ready | 05:48:18 | 05:49:56 | 98s | Time from uvicorn start message to models loaded |
| API model load phase | 05:49:52 | 05:49:56 | 4s | From "Starting DeepFake Detector API..." to "Models loaded successfully!" |

### Phase 2 Observations

- Submodel loading now overlaps in runtime logs (bounded parallel local initialization is active).
- Runtime API model load phase improved slightly (5s -> 4s).
- End-to-end startup remained dominated by pre-lifespan/init time (98s still much larger than model load slice).
- Runtime hygiene warnings no longer appeared in this run (no OMP warning and no sklearn pickle version warning).

## Phase 3 Results and Bottleneck Attribution

Source log window:
- Build queued at 2026-04-20 06:01:57
- Application startup begins at 2026-04-20 06:02:56
- Models loaded successfully at 2026-04-20 06:04:37

### Phase 3 Timing Summary

| Segment | Start | End | Duration | Notes |
|---|---:|---:|---:|---|
| Queue/build to app startup | 06:01:57 | 06:02:56 | 59s | Includes scheduling, build finalization, image start |
| App startup to model-ready | 06:02:56 | 06:04:37 | 101s | End-to-end startup from Space startup marker |
| API model load phase | 06:04:33 | 06:04:37 | 4s | From app startup handler to models loaded |

### Phase 3 Instrumentation Breakdown (Container Runtime)

| Marker | Duration |
|---|---:|
| module_import_complete | 4.050s |
| startup_model_load_duration_seconds | 3.967s |
| startup_lifespan_total_duration_seconds | 3.967s |
| load_from_fusion_repo_total_duration_seconds | 3.967s |

### Bottleneck Attribution

- Dominant gap is before module import:
    - 06:02:56 (Application Startup) -> 06:04:29 (module_import_start) = 93s.
- App code after import is no longer the main problem:
    - import + lifespan + model load is about 8s total.
- Conclusion:
    - Remaining cold start is primarily platform/container readiness overhead, not model download/load logic.

## Comparison Template (Fill After Implementation)

| Metric | Baseline (2026-04-20) | After Phase 1 | After Phase 2 | After Phase 3 | Final |
|---|---:|---:|---:|---:|---:|
| Queue/build to app startup | 28s | 36s | 119s | 59s |  |
| App startup to model-ready | 94s | 99s | 98s | 101s |  |
| API model load phase | 21s | 5s | 4s | 4s |  |
| vit-base load | 13s | 1s | 2s | 2s |  |
| deit-distilled load | 5s | 2s | 2s | 2s |  |
| Total visible build timed stages | 20.4s | 28.0s | 112.7s | 33.6s |  |
| Phase3 module import duration | n/a | n/a | n/a | 4.050s |  |
| Phase3 model registry total duration | n/a | n/a | n/a | 3.967s |  |

## Expected Outcome

Primary expected wins:
1. Reduced startup latency by avoiding runtime model downloads.
2. Reduced model load wall-clock via parallel submodel loads.
3. Stable and comparable timing data for iterative tuning.

Secondary expected wins:
1. Cleaner startup logs (no OMP warning).
2. Lower risk from sklearn deserialization mismatch.

## Rollback Plan

If anything regresses:
1. Revert parallel loading only and keep build-time prefetch.
2. Revert build-time prefetch and restore runtime download flow.
3. Keep instrumentation to retain comparability.

## Notes

- This plan intentionally keeps current FastAPI inference architecture unchanged.
- Triton feasibility can be revisited after cold start metrics improve and stabilize.
