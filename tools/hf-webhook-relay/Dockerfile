FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=7860

RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user:user app.py .

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
