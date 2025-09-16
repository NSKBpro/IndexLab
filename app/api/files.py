import uuid, asyncio
from datetime import datetime
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from ..core.config import settings
from ..ingest.schema import IngestConfig
from ..models.db import Job, get_session
from ..ingest.pipeline import run_pipeline

router = APIRouter()

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    index_name: str = Form(...),
    text_column: str | None = Form(None),
    embedding_model: str = Form(None),
    normalize_embeddings: bool = Form(True),
    chunk_mode: str = Form("fixed_chars"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
    backend: str = Form("faiss_flat"),
    nlist: int = Form(1024),
    nprobe: int = Form(10),
    M: int = Form(16),
    ef_construction: int = Form(200),
    ef_search: int = Form(64),
):
    if not index_name.strip(): raise HTTPException(400, "index_name required")
    model = (embedding_model or settings.default_model).strip()
    if model not in settings.allowed_models:
        raise HTTPException(400, f"embedding_model not allowed: {model}")
    if backend not in settings.allowed_backends:
        raise HTTPException(400, f"backend not allowed: {backend}")

    job_id = str(uuid.uuid4())
    dest = settings.uploads_dir / f"{job_id}_{file.filename}"
    with dest.open("wb") as f:
        f.write(await file.read())

    with get_session() as s:
        s.add(Job(id=job_id, status="queued", source_filename=file.filename)); s.commit()

    cfg = IngestConfig(
        index_name=index_name.strip(),
        text_column=text_column,
        embedding_model=model,
        normalize_embeddings=bool(normalize_embeddings),
        chunk_mode=chunk_mode, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap),
        backend=backend, nlist=int(nlist), nprobe=int(nprobe), M=int(M),
        ef_construction=int(ef_construction), ef_search=int(ef_search),
    )
    asyncio.create_task(run_pipeline(job_id, dest, cfg))
    return {"job_id": job_id}

@router.get("/status/{job_id}")
def get_status(job_id: str):
    with get_session() as s:
        j = s.get(Job, job_id)
        if not j: raise HTTPException(404, "job not found")
        return {"id": j.id, "status": j.status, "message": j.message, "index_name": j.index_name, "src": j.source_filename}
