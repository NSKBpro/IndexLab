# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends     build-essential gcc g++ libopenblas-dev libomp-dev     && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --upgrade pip && pip install -e .

COPY app ./app

RUN mkdir -p /data
ENV DATA_DIR=/data UPLOADS_DIR=/data/uploads INDEXES_DIR=/data/indexes DB_PATH=/data/app.sqlite     CHARTS_DIR=/data/charts EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2     ALLOWED_MODELS="sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-m3, intfloat/e5-base-v2"     DEFAULT_BACKEND=faiss_flat TOP_K=5 CHUNK_MODE=fixed_chars CHUNK_SIZE=1000 CHUNK_OVERLAP=150

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host=0.0.0.0", "--port=8000", "--workers=2"]
