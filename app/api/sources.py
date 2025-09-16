from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from ..core.config import settings

router = APIRouter()

def _clean_index_name(filename: str) -> str:
    # filename is like "foo.faiss.manifest.json" or "foo.manifest.json"
    if filename.endswith(".faiss.manifest.json"):
        return filename[: -len(".faiss.manifest.json")]
    if filename.endswith(".manifest.json"):
        return filename[: -len(".manifest.json")]
    # fallback: strip last .json only
    if filename.endswith(".json"):
        return filename[:-5]
    return filename

def _manifest_path_for(index_name: str) -> Path | None:
    """Return the actual manifest path for a given logical index name."""
    d = settings.indexes_dir
    candidates = [
        d / f"{index_name}.faiss.manifest.json",
        d / f"{index_name}.manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # last resort: scan directory in case name has odd casing or whitespace
    for mf in d.glob("*.manifest.json"):
        clean = _clean_index_name(mf.name)
        if clean == index_name:
            return mf
    return None

def _iter_manifests():
    for mf in settings.indexes_dir.glob("*.manifest.json"):
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
            yield _clean_index_name(mf.name), manifest, mf
        except Exception:
            continue

@router.get("/indexes")
def list_indexes():
    out = []
    for name, man, _ in _iter_manifests():
        out.append({
            "index_name": name,
            "count": man.get("count"),
            "model": man.get("model"),
            "backend": man.get("backend"),
            "created_at": man.get("created_at"),
            "chunking": man.get("chunking", {}),
            "has_sources": bool(man.get("sources")),
        })
    out.sort(key=lambda x: x["index_name"])
    return {"indexes": out}

@router.get("/sources/{index_name}")
def get_sources(index_name: str):
    mf = _manifest_path_for(index_name)
    if mf is None:
        raise HTTPException(404, f"index manifest not found for '{index_name}'")
    man = json.loads(mf.read_text(encoding="utf-8"))
    sources = man.get("sources") or {}
    return {
        "index_name": index_name,
        "count": man.get("count", 0),
        "model": man.get("model"),
        "backend": man.get("backend"),
        "created_at": man.get("created_at"),
        "chunking": man.get("chunking", {}),
        "sources": sources,
    }
