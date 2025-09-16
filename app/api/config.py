from fastapi import APIRouter
from ..core.config import settings

router = APIRouter()

@router.get("/config")
def get_config():
    return {
        "allowed_models": settings.allowed_models,
        "allowed_backends": settings.allowed_backends,
        "defaults": {
            "embedding_model": settings.default_model,
            "normalize_embeddings": settings.normalize_embeddings,
            "chunk_mode": settings.chunk_mode,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "backend": settings.default_backend,
            "top_k": settings.top_k
        }
    }
