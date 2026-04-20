# DeepFake Detector API - Docker Image
# Unified Dockerfile for local, Railway, and HF Spaces

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user (HF Spaces requirement)
RUN useradd -m -u 1000 user
USER user

# Set PATH for user-installed packages
ENV PATH="/home/user/.local/bin:$PATH"

# Copy requirements and install dependencies as user
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy application code
COPY --chown=user:user . /app

# Switch to root to create cache directory and set permissions
USER root
RUN mkdir -p /app/.hf_cache && chown -R user:user /app/.hf_cache && chmod +x /app/start.sh

# Switch back to user
USER user

# Expose default app port
EXPOSE 7860

# Run the application (start.sh uses PORT env var)
CMD ["./start.sh"]
