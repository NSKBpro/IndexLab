from pydantic import BaseModel
from pathlib import Path
import os

class Settings(BaseModel):
    data_dir: Path = Path(os.getenv("DATA_DIR", "data")).resolve()
    uploads_dir: Path = Path(os.getenv("UPLOADS_DIR", "data/uploads")).resolve()
    indexes_dir: Path = Path(os.getenv("INDEXES_DIR", "data/indexes")).resolve()
    charts_dir: Path = Path(os.getenv("CHARTS_DIR", "data/charts")).resolve()
    db_path: Path = Path(os.getenv("DB_PATH", "data/app.sqlite")).resolve()

    allowed_models: list[str] = [m.strip() for m in os.getenv(
        "ALLOWED_MODELS",
        "sentence-transformers/all-MiniLM-L6-v2, BAAI/bge-m3, intfloat/e5-base-v2"
    ).split(",") if m.strip()]
    default_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    normalize_embeddings: bool = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"

    chunk_mode: str = os.getenv("CHUNK_MODE", "fixed_chars")  # fixed_chars|sentences|headings
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    top_k: int = int(os.getenv("TOP_K", "5"))

    allowed_backends: list[str] = ["faiss_flat", "faiss_ivf"]
    default_backend: str = "faiss_flat"


settings = Settings()
for p in (settings.data_dir, settings.uploads_dir, settings.indexes_dir, settings.charts_dir):
    p.mkdir(parents=True, exist_ok=True)
