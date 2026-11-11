# DeepFake Detector API - App Image
# Uses a prebuilt dependency base image when provided, with fallback install path.

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860 \
    HF_CACHE_DIR=/app/.hf_cache \
    HF_HUB_CACHE=/app/.hf_cache \
    HF_HOME=/app/.hf_cache/hf_home \
    HF_XET_CACHE=/app/.hf_cache/xet

ARG INSTALL_OPTIONAL_DEPS=false
USER root
RUN id -u user >/dev/null 2>&1 || useradd -m -u 1000 user
COPY --chown=user:user requirements-torch.txt requirements-runtime.txt requirements-optional.txt requirements.txt ./

# If the base image already contains runtime deps, this check passes and install is skipped.
# Fallback path: pinned CUDA torch from the PyTorch index first, then the rest
# constrained to that torch (never --upgrade, which would pull PyPI torch).
RUN python -c "import fastapi, torch, timm, sklearn, google.genai" \
    || ( pip install --no-cache-dir -r requirements-torch.txt \
         && pip freeze | grep -E '^(torch|torchvision)==' > /tmp/torch-constraints.txt \
         && pip install --no-cache-dir -r requirements-runtime.txt -c /tmp/torch-constraints.txt \
         && rm -rf /root/.cache /tmp/torch-constraints.txt )

RUN if [ "$INSTALL_OPTIONAL_DEPS" = "true" ]; then \
         python -c "import pytest" \
         || pip install --no-cache-dir --upgrade -r requirements-optional.txt; \
    fi

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy only files required for model prefetch first.
COPY --chown=user:user app /app/app
# model-manifest.json is optional: the wildcard lets the build pass when it is
# absent (start.sh always matches, so the COPY never has zero sources).
COPY --chown=user:user start.sh model-manifest.json* /app/

# Switch to root to create cache directory and set permissions
USER root
RUN mkdir -p /app/.hf_cache /app/.hf_cache/hf_home /app/.hf_cache/xet \
    && chown -R user:user /app/.hf_cache \
    && chmod +x /app/start.sh

# Switch back to user
USER user

# Expose default app port
EXPOSE 7860

# Run the application (start.sh uses PORT env var)
CMD ["./start.sh"]
