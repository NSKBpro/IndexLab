# app/ingest/pipeline.py
import hashlib
import json, logging, re, shutil
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
    name = Path(p).name
    m = re.match(r'^[0-9a-fA-F-]{36}[_-](.+)$', name)
    return m.group(1) if m else name


def chunk_text(text: str, mode: str, size: int, overlap: int) -> list[str]:
    if mode == "fixed_chars":
        return chunk_fixed(text, size, overlap)
    if mode == "sentences":
        return chunk_sentences(text, size, overlap)
    if mode == "headings":
        return chunk_by_headings(text, size, overlap)
    return chunk_fixed(text, size, overlap)


async def run_pipeline(job_id: str, file_path: Path, cfg) -> None:
    try:
        # mark job running
        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "running"
            job.updated_at = datetime.utcnow()
            s.add(job)
            s.commit()

        # 1) Read + normalize
        await event_bus.publish(job_id, "Reading file")
        df = normalize_df(read_any(file_path))

        # 2) Chunk
        await event_bus.publish(job_id, "Chunking")
        pairs: list[tuple[str, str]] = []
        for doc_id, text in iter_rows(df, cfg.text_column):
            for j, chunk in enumerate(
                chunk_text(text, cfg.chunk_mode, cfg.chunk_size, cfg.chunk_overlap)
            ):
                pairs.append((f"{doc_id}#{j}", chunk))

        ids = [p[0] for p in pairs]
        texts = [p[1] for p in pairs]

        # 3) Embed
        await event_bus.publish(job_id, f"Embedding {len(texts)} with {cfg.embedding_model}")
        embs = embed_texts(texts, cfg.embedding_model, cfg.normalize_embeddings)

        # 4) Write docs (LATEST)
        await event_bus.publish(job_id, f"Building index [{cfg.backend}]")
        index_path = settings.indexes_dir / f"{cfg.index_name}.faiss"
        docs_path = index_path.with_suffix(".docs.json")
        docs_path.write_text(json.dumps(dict(pairs), ensure_ascii=False), encoding="utf-8")

        # 5) Manifest (LATEST)
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
                "overlap": cfg.chunk_overlap,
            },
            "metric": "ip",
            "created_at": datetime.utcnow().isoformat(timespec="seconds"),
            "sources": source_rec,
        }

        latest_manifest_path = index_path.with_suffix(".manifest.json")
        latest_manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        # 6) Build vector index
        build_index(embs, ids, manifest, index_path)

        # 7) Version archival (TRUE per-version)
        version = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        versions_root = settings.indexes_dir / cfg.index_name / "versions"
        version_dir = versions_root / version
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts into version folder
        shutil.copy2(index_path,            version_dir / f"{cfg.index_name}.faiss")
        shutil.copy2(docs_path,             version_dir / f"{cfg.index_name}.docs.json")

        # Write BOTH names for manifest to satisfy loaders:
        #   - manifest.json
        #   - <name>.manifest.json   (what load_index(<faiss_path>) will look for)
        (version_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (version_dir / f"{cfg.index_name}.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # If your build_index writes an .ids.json beside latest, copy it too (optional but helpful)
        ids_json = index_path.with_suffix(".ids.json")
        if ids_json.exists():
            shutil.copy2(ids_json, version_dir / f"{cfg.index_name}.ids.json")

        # Meta (for listing)
        def _sum_rows(sources: dict) -> int | None:
            try:
                return int(sum(int(v.get("rows", 0)) for v in sources.values()))
            except Exception:
                return None

        index_version = {
            "version": version,
            "created_at": manifest["created_at"],
            "embed_model": manifest["model"],
            "chunking": manifest["chunking"]["mode"],
            "chunk_size": manifest["chunking"]["size"],
            "chunk_overlap": manifest["chunking"]["overlap"],
            "index_backend": manifest["backend"],
            "doc_count": _sum_rows(source_rec),
            "vector_count": manifest["count"],
            "build_id": job_id,
            "notes": f"Built from {Path(file_path).name}",
            "metrics": {"recall@k": None, "mrr": None, "ndcg": None},
        }
        (version_dir / "meta.json").write_text(json.dumps(index_version, ensure_ascii=False, indent=2), encoding="utf-8")
        (versions_root / f"{version}.json").write_text(json.dumps(index_version, ensure_ascii=False, indent=2), encoding="utf-8")

        # mark job done
        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "done"
            job.index_name = cfg.index_name
            job.updated_at = datetime.utcnow()
            s.add(job)
            s.commit()

        await event_bus.publish(job_id, "DONE")

    except Exception as e:
        log.exception("Pipeline error")
        with get_session() as s:
            job = s.get(Job, job_id)
            job.status = "error"
            job.message = str(e)
            job.updated_at = datetime.utcnow()
            s.add(job)
            s.commit()
        await event_bus.publish(job_id, f"ERROR: {e}")
