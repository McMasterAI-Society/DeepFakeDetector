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
COPY --chown=user:user requirements-runtime.txt requirements-optional.txt requirements.txt ./

# If the base image already contains runtime deps, this check passes and install is skipped.
RUN python -c "import fastapi, torch, timm, sklearn, google.genai" \
    || pip install --no-cache-dir --upgrade -r requirements-runtime.txt

RUN if [ "$INSTALL_OPTIONAL_DEPS" = "true" ]; then \
         python -c "import pytest" \
         || pip install --no-cache-dir --upgrade -r requirements-optional.txt; \
    fi

USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Copy only files required for model prefetch first.
COPY --chown=user:user app /app/app
COPY --chown=user:user start.sh /app/start.sh
COPY --chown=user:user model-manifest.json /app/model-manifest.json

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
