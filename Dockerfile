# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Runtime libs only (no compilers). libgomp1 satisfies OpenMP symbols many ML wheels need.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only manifests first for better cache
COPY pyproject.toml /app/

# Install project deps from pyproject (non-editable, no source build)
# If you use Poetry, replace with poetry install (see note below).
RUN pip install --upgrade pip \
 && pip install .    # IMPORTANT: not "-e ."

# Copy your app code
COPY app /app/app

# Data dirs & defaults
RUN mkdir -p /data
ENV DATA_DIR=/data \
    UPLOADS_DIR=/data/uploads \
    INDEXES_DIR=/data/indexes \
    DB_PATH=/data/app.sqlite \
    CHARTS_DIR=/data/charts \
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 \
    ALLOWED_MODELS="sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-m3, intfloat/e5-base-v2" \
    DEFAULT_BACKEND=faiss_flat \
    TOP_K=5 \
    CHUNK_MODE=fixed_chars \
    CHUNK_SIZE=1000 \
    CHUNK_OVERLAP=150 \
    # Optional: keep HF caches on a Railway Volume mounted at /data
    HF_HOME=/data/hf \
    TRANSFORMERS_CACHE=/data/transformers

# Railway provides $PORT. Keep workers to 1 to avoid OOM on free tier.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
