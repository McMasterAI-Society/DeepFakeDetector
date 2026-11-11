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

## Phase 8: Image Size, Early Wake-up, Warm-up and Inference Path

The Phase 3 attribution shows ~93 s between the Space "Application startup"
marker and `phase3 module_import_start`, against ~8 s for import + model
load. That gap is container pull/start, so this phase targets image size
first, then hides what remains behind the user's first actions, then trims
the first-request and steady-state inference cost.

All numbers in the results tables below are **TBD**: they must be measured on
the real Space with the protocol at the end of this section.

### 8.1 Slim dependency image (CS2)

Files: `Dockerfile.base`, `Dockerfile`, `requirements-torch.txt`, `.dockerignore`.

- `Dockerfile.base` is now multi-stage. The builder installs everything into
  `/opt/venv`; the runtime stage is a clean `python:3.11-slim` plus one
  `COPY --from=builder /opt/venv` layer. No pip cache, no build leftovers, no
  layers holding files that were later deleted.
- torch / torchvision come from the official PyTorch CUDA index, pinned in
  `requirements-torch.txt` (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`).
  The rest of `requirements-runtime.txt` is installed with a constraint on the
  resolved torch so it is never re-resolved from PyPI.
  - Why pin: unpinned PyPI `torch>=2.0,<3.0` currently resolves to a CUDA 13
    build (observed during a local build of the old `Dockerfile.base`:
    `torch-2.15.0` with `nvidia-*-cu13` wheels). CUDA 13 wheels need a newer
    NVIDIA driver; cu124 needs driver >= 525.60 and includes sm_75 (T4).
    Rebuilding the old base image could silently produce an image that does
    not use the T4.
- `SLIM_TORCH=true` (build arg, default) removes `triton` (only used by
  `torch.compile`/inductor), `torch/include`, `torch/share/cmake`, the
  `nvidia/*/include` header dirs and static `*.a` archives. A build-time
  `python -c "import torch, torchvision, timm, ..."` check fails the build if
  slimming broke anything. Use `--build-arg SLIM_TORCH=false` if
  `torch.compile` is ever enabled (CS7).
- `.pyc` files are kept on purpose: with `PYTHONDONTWRITEBYTECODE=1` nothing is
  ever cached at runtime, so stripping them would make every cold import
  recompile (slower `module_import_complete`).
- Kept unchanged: GHCR base-image flow (`update-ghcr-base.*`, `BASE_IMAGE`
  arg rewritten by `deploy-to-hf.sh`), uid 1000 `user`, `/app/.hf_cache` and
  the `/data` persistent cache logic in `start.sh`.
- The app `Dockerfile` fallback path (no prebuilt base) uses the same pinned
  torch install instead of `pip install --upgrade` from PyPI.
- The app `Dockerfile` also copies `model-manifest.json` into `/app/` when it
  exists (`COPY start.sh model-manifest.json* /app/`; the build does not fail
  when the manifest is absent).
- Not done here: a CUDA-runtime base image (`nvidia/cuda:*-runtime`) was not
  used because the pip `nvidia-*` wheels already provide the CUDA libraries;
  a CUDA base would add them a second time. `matplotlib` is still installed
  because `app/services/explainability.py` imports it for colormaps; dropping
  it needs a change in the explainability code.

Expected effect (estimate from wheel sizes, not a measured image size).
Compressed download sizes of the torch stack:

| Stack | torch wheel | nvidia-* wheels | triton | Total |
|---|---:|---:|---:|---:|
| Unpinned PyPI today (torch 2.15.0, cu13) | 539 MB | ~2,165 MB | 286 MB | ~2,990 MB |
| Pinned torch 2.6.0+cu124 | 769 MB | ~2,030 MB | 253 MB | ~3,050 MB |
| Pinned cu124 with `SLIM_TORCH=true` | 769 MB | ~2,030 MB | removed | ~2,800 MB |

(Sizes from the PyTorch index / PyPI file listings, 2026-11. Installed,
uncompressed sizes are larger. No image was built locally for this
comparison.)

So the index switch is **not** a size win by itself: modern PyTorch CUDA
wheels on both PyPI and download.pytorch.org pull the same kind of `nvidia-*`
pip wheels (cuDNN alone is 520 to 665 MB). The expected size reduction comes
from removing triton, headers and static archives, and from the multi-stage
copy: roughly 8 to 10 % of the torch stack, or about 6 % less than today's
unpinned cu13 stack. The WS-5 target of "under 50 %
of the current size" is **not** expected from these changes. Getting there
would need dropping CUDA libraries that libtorch links (not safe without
testing) or a different runtime (ONNX Runtime / TensorRT, CS7).
The main value of the pin is correctness: the T4 keeps a cu12x build, and
the torch version stays fixed between rebuilds. Before deploying, compare
the pin with `pip freeze | grep torch` from the current GHCR base image and
run the golden-output test (WS-1).

### 8.2 Wake the Space early (CS3)

Files: `frontend/src/lib/backendWarmup.ts`, `frontend/src/hooks/use-backend-ready.ts`,
`frontend/src/components/BackendWarmupBanner.tsx`, `frontend/src/main.tsx`,
`frontend/src/pages/Analyze.tsx` (one line: the banner).

- On app load (`main.tsx`) the frontend sends one fire-and-forget
  `GET {VITE_API_BASE_URL}/health` per browser session (sessionStorage guard).
  It is a simple GET with no custom headers, so there is no CORS preflight.
- On Analyze, `/ready` is polled every 3 s (10 s timeout per call, stops after
  10 min). While it is not ready, a small "Warming up the GPU server… N s"
  banner is shown (hidden for the first 2 s to avoid flicker on a warm
  Space). Submitting is never blocked.
- `/ready` semantics (`routes_health.py`, unchanged): `status="ready"` once
  the model registry is loaded. Because uvicorn only starts serving after the
  lifespan startup finishes, `/ready` answers only after model load and the
  warm-up below, so "ready" means "first request will be warm".

Expected effect (expectation): for a visitor who lands on Home and then goes
to Analyze, most of the cold start overlaps with reading the page and choosing
an image.

### 8.3 Startup warm-up (CS5)

Files: `app/main.py`, `app/services/inference_service.py`, `app/core/config.py`.

- After models load, the lifespan runs `warm_up_on_startup()` in a worker
  thread: it sets `torch.backends.cudnn.benchmark = True` on CUDA
  (`CUDNN_BENCHMARK`, default true), then runs one non-explain forward per
  submodel on a deterministic 256x256 noise image.
- Settings: `WARMUP_ON_STARTUP` (default `true`), `CUDNN_BENCHMARK` (default
  `true`). Warm-up failures are logged and never stop startup.
- New log markers (same `phase3` style):
  - `phase3 cudnn_benchmark=enabled`
  - `phase3 startup_warmup_model name=<model> duration_seconds=<s>`
  - `phase3 startup_warmup_duration_seconds=<s> models=<n>`
  - `phase3 startup_warmup skipped reason=disabled|models_not_loaded`
- The warm-up uses `explain=False`. The explain path's backward kernels still
  initialise on the first explain request.

Expected effect (expectation): startup grows by roughly the warm-up time;
the first user request no longer pays CUDA context / kernel load / cuDNN
autotune (WS-5 estimate: 1 to 3 s).

Local check (RTX 4060 Laptop GPU, Windows, torch 2.9.1+cu128; cnn-transfer,
gradfield-cnn and deit-distilled only, ViT not loaded). This is **not** a T4
number:

| | warm-up total | cnn-transfer warm-up |
|---|---:|---:|
| `CUDNN_BENCHMARK=true` | ~26 s | ~25 s |
| `CUDNN_BENCHMARK=false` | ~1.1 s | ~0.7 s |

On this machine cuDNN autotuning for the EfficientNet backbone dominates the
warm-up. Steady-state explain-off latency was lower with benchmark on, but the
laptop numbers were noisy. **Decide the default from T4 data**: if
`startup_warmup_duration_seconds` on the Space is similar, set
`CUDNN_BENCHMARK=false` as a Space variable; the extra startup would cost
more than the per-request saving.

### 8.4 Inference path (CS6)

File: `app/services/inference_service.py` (wrappers unchanged).

- The image is decoded once per request (`Image.open(...).convert("RGB")`,
  identical to what each wrapper did) and the same PIL image is passed to all
  four submodels via `image=`. Raw bytes are still passed for bytes-only
  wrappers (dummy models). A decode failure now raises `ImageProcessingError`
  (HTTP 400) instead of four wrapped `InferenceError`s.
- `explain=false` calls run under `torch.inference_mode()`. The explain path
  (Grad-CAM, attention rollout) runs exactly as before, with autograd.
- `INFERENCE_FP16` (default `false`): fp16 autocast on CUDA for the
  non-explain path only. Off by default because it changes probabilities
  slightly; validate against the golden outputs before enabling.
- Note: the frontend currently always calls `/predict?explain=true`, so
  `inference_mode`/fp16 only help API users and a future fast-path
  (`/predict` without explain, explain fetched lazily). Decode-once helps both.

Expected effect (expectation): small CPU savings from 3 fewer decodes per
request (largest for big JPEG/PNG uploads); `inference_mode` gives a small
reduction on explain-off requests.

Local check (same machine and 3 submodels as 8.3, 5 demo images, then 30
interleaved runs on one image; not a T4 number):
- `prob_fake` from the old path (each wrapper decodes, no `inference_mode`) and
  the new path were identical (max abs difference 0.0), explain off and on.
- Explain off p50: 112 ms old vs 99 ms new (benchmark on); 192 ms vs 180 ms
  (benchmark off). Explain on: about the same (610 vs 601 ms; 608 vs 587 ms).

### 8.5 Measurement protocol

Tools: `scripts/bench_latency.py` (repo root) and the Space logs.

1. Image size (CS1): build both base images locally and record
   `docker images` size, and the compressed size from `docker manifest inspect`
   or the GHCR package page after a push.
   ```bash
   docker build -f backend/Dockerfile.base -t dfd-base:slim backend
   docker images dfd-base
   ```
2. Cold start, 5 runs per variant: put the Space to sleep (or restart it),
   then immediately run
   ```bash
   python scripts/bench_latency.py --base-url https://<user>-<space>.hf.space \
       --cold --n 20 --label <variant> --out bench-<variant>-<run>.json
   ```
   This records `cold_start.time_to_first_response_s`,
   `cold_start.time_to_ready_s` and `first_request_after_ready` (explain off).
   From the Space logs, also record Application startup -> `module_import_start`
   and the `phase3 startup_warmup_*` markers.
3. Warm latency: the same run reports `/predict` p50/p95 (client wall time and
   server `timing_ms`) for explain off and explain on.
4. Output check: `prob_fake_values` in the JSON must match between variants
   with `INFERENCE_FP16=false` (same image, same models).
5. Fill in the tables below. Report medians of the 5 runs.

### Phase 8 Results (TBD)

| Metric | Before (current main) | After Phase 8 |
|---|---:|---:|
| Base image size, uncompressed (`docker images`) | TBD | TBD |
| Base image size, compressed (registry) | TBD | TBD |
| Application startup -> module_import_start | 93s (Phase 3) | TBD |
| time_to_ready_s (bench, median of 5) | TBD | TBD |
| startup_warmup_duration_seconds | n/a | TBD |
| First request after ready, explain off (ms) | TBD | TBD |
| Warm /predict p50 / p95, explain off (ms) | TBD | TBD |
| Warm /predict p50 / p95, explain on (ms) | TBD | TBD |

## Rollback Plan

If anything regresses:
1. Revert parallel loading only and keep build-time prefetch.
2. Revert build-time prefetch and restore runtime download flow.
3. Keep instrumentation to retain comparability.

## Notes

- This plan intentionally keeps current FastAPI inference architecture unchanged.
- Triton feasibility can be revisited after cold start metrics improve and stabilize.
