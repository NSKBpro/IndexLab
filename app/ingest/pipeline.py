import hashlib
import json, logging
import re
from datetime import datetime
from pathlib import Path

from ..core.config import settings
from ..core.sse import event_bus
from ..models.db import Job, get_session
from .reader import read_any
from .normalize import normalize_df
from .chunker import iter_rows, chunk_fixed, chunk_sentences, chunk_by_headings
from .embedder import embed_texts
from .indexer import build_index

log = logging.getLogger(__name__)

def _pretty_source_name(p: Path | str) -> str:
    """
    Return a nice display name for a source file.
    Strips a leading UUID_ or UUID- prefix if present, otherwise returns the basename.
    """
    name = Path(p).name
    m = re.match(r'^[0-9a-fA-F-]{36}[_-](.+)$', name)
    return m.group(1) if m else name

def chunk_text(text: str, mode: str, size: int, overlap: int) -> list[str]:
    if mode == "fixed_chars": return chunk_fixed(text, size, overlap)
    if mode == "sentences":   return chunk_sentences(text, size, overlap)
    if mode == "headings":    return chunk_by_headings(text, size, overlap)
    return chunk_fixed(text, size, overlap)

async def run_pipeline(job_id: str, file_path: Path, cfg) -> None:
    try:
        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "running"; job.updated_at = datetime.utcnow()
            s.add(job); s.commit()

        await event_bus.publish(job_id, "Reading file")
        df = normalize_df(read_any(file_path))

        await event_bus.publish(job_id, "Chunking")
        pairs = []
        for doc_id, text in iter_rows(df, cfg.text_column):
            for j, chunk in enumerate(chunk_text(text, cfg.chunk_mode, cfg.chunk_size, cfg.chunk_overlap)):
                pairs.append((f"{doc_id}#{j}", chunk))

        ids   = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]

        await event_bus.publish(job_id, f"Embedding {len(texts)} with {cfg.embedding_model}")
        embs = embed_texts(texts, cfg.embedding_model, cfg.normalize_embeddings)

        await event_bus.publish(job_id, f"Building index [{cfg.backend}]")
        index_path = settings.indexes_dir / f"{cfg.index_name}.faiss"
        index_path.with_suffix(".docs.json").write_text(
            json.dumps(dict(pairs), ensure_ascii=False), encoding="utf-8"
        )

        # --- compute source metadata for manifest ---
        try:
            raw = file_path.read_bytes()
            sha256 = hashlib.sha256(raw).hexdigest()
        except Exception:
            sha256 = None

        source_rec = {
            _pretty_source_name(file_path): {
                "rows": int(len(df)),
                "sha256": sha256,
                "added_at": datetime.utcnow().isoformat(timespec="seconds"),
                "stored_name": Path(file_path).name,
            }
        }

        manifest = {
            "dim": int(embs.shape[1]),
            "count": len(ids),
            "model": cfg.embedding_model,
            "normalize": bool(cfg.normalize_embeddings),
            "backend": cfg.backend,
            "params": {
                "nlist": cfg.nlist,
                "nprobe": cfg.nprobe,
                "M": cfg.M,
                "efConstruction": cfg.ef_construction,
                "efSearch": cfg.ef_search,
            },
            "chunking": {
                "mode": cfg.chunk_mode,
                "size": cfg.chunk_size,
                "overlap": cfg.chunk_overlap
            },
            "metric": "ip",
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            "sources": source_rec,
        }

        build_index(embs, ids, manifest, index_path)

        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "done"; job.index_name = cfg.index_name; job.updated_at = datetime.utcnow()
            s.add(job); s.commit()

        await event_bus.publish(job_id, "DONE")

    except Exception as e:
        log.exception("Pipeline error")
        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "error"; job.message = str(e); job.updated_at = datetime.utcnow()
            s.add(job); s.commit()
        await event_bus.publish(job_id, f"ERROR: {e}")
