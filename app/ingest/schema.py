from pydantic import BaseModel

class IngestConfig(BaseModel):
    index_name: str
    text_column: str | None = None
    embedding_model: str
    normalize_embeddings: bool
    # chunking
    chunk_mode: str
    chunk_size: int
    chunk_overlap: int
    # backend
    backend: str
    # IVF
    nlist: int = 1024
    nprobe: int = 10
    # HNSW
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 64
