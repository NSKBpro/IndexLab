# app/api/sources_api.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Query

from ..core.config import settings

# NOTE: main.py should include this router as:
# app.include_router(router, prefix="/api")
router = APIRouter()


# ----------------------------- helpers -----------------------------

def _clean_index_name(filename: str) -> str:
    """
    Convert a manifest filename like
      foo.faiss.manifest.json  -> foo
      foo.manifest.json        -> foo
      fallback: strip trailing .json
    to a logical index name.
    """
    if filename.endswith(".faiss.manifest.json"):
        return filename[: -len(".faiss.manifest.json")]
    if filename.endswith(".manifest.json"):
        return filename[: -len(".manifest.json")]
    if filename.endswith(".json"):
        return filename[:-5]
    return filename


def _manifest_path_latest(index_name: str) -> Optional[Path]:
    """
    Find the 'latest' manifest for an index (no version specified).
    Checks:
      <indexes_dir>/<name>.faiss.manifest.json
      <indexes_dir>/<name>.manifest.json
    Falls back to scanning *.manifest.json and matching by cleaned name.
    """
    d = settings.indexes_dir
    candidates = [
        d / f"{index_name}.faiss.manifest.json",
        d / f"{index_name}.manifest.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    # last resort: scan directory (for odd casing/whitespace)
    for mf in d.glob("*.manifest.json"):
        if _clean_index_name(mf.name) == index_name:
            return mf
    return None


def _manifest_path_version(index_name: str, version: str) -> Optional[Path]:
    """
    Versioned manifest lives at:
      <indexes_dir>/<name>/versions/<version>/manifest.json

    If not found, returns None.
    (We deliberately do NOT use legacy versions/<version>.json here,
     because that file is 'meta', not a manifest with sources.)
    """
    vpath = settings.indexes_dir / index_name / "versions" / version / "manifest.json"
    return vpath if vpath.exists() else None


def _iter_latest_manifests() -> Iterator[Tuple[str, Dict[str, Any], Path]]:
    """
    Iterate all 'latest' manifests in the root directory.
    Yields (index_name, manifest_dict, file_path).
    """
    for mf in settings.indexes_dir.glob("*.manifest.json"):
        try:
            manifest = json.loads(mf.read_text(encoding="utf-8"))
            yield _clean_index_name(mf.name), manifest, mf
        except Exception:
            # skip malformed
            continue


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Invalid JSON at {path.name}: {e}")


# ------------------------------- routes -------------------------------

@router.get("/indexes")
def list_indexes():
    """
    Lists ONLY 'latest' indexes (based on *.manifest.json in the root).
    If you also need a full version listing, use your /api/versioning endpoints.
    """
    out = []
    for name, man, _ in _iter_latest_manifests():
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
def get_sources(
    index_name: str,
    version: Optional[str] = Query(default=None, description="Version id (e.g. 20250917-091032). Omit for latest."),
):
    """
    Returns sources (per-file metadata) from the manifest for either:
      - a specific version (versions/<version>/manifest.json), or
      - the latest manifest beside the latest index files.

    Response matches your existing shape so the UI doesn't need to change.
    """
    # Resolve manifest path (versioned vs latest)
    if version:
        mf = _manifest_path_version(index_name, version)
        if mf is None:
            raise HTTPException(404, f"manifest not found for '{index_name}' version '{version}'")
    else:
        mf = _manifest_path_latest(index_name)
        if mf is None:
            raise HTTPException(404, f"index manifest not found for '{index_name}'")

    man = _read_json(mf)
    sources = man.get("sources") or {}

    # Helpful debug flags (optional: remove if you don't want them)
    base = settings.indexes_dir / index_name
    latest_faiss = settings.indexes_dir / f"{index_name}.faiss"
    latest_docs = latest_faiss.with_suffix(".docs.json")
    latest_manifest = latest_faiss.with_suffix(".manifest.json")

    vdir = base / "versions" / version if version else None
    v_faiss = (vdir / f"{index_name}.faiss") if vdir else None
    v_docs = (vdir / f"{index_name}.docs.json") if vdir else None
    v_manifest = (vdir / "manifest.json") if vdir else None

    return {
        "index_name": index_name,
        "version": version,
        "count": man.get("count", 0),
        "model": man.get("model"),
        "backend": man.get("backend"),
        "created_at": man.get("created_at"),
        "chunking": man.get("chunking", {}),
        "sources": sources,
        "dim": man.get("dim"),
        # debug presence flags
        "has_faiss_latest": latest_faiss.exists(),
        "has_docs_latest": latest_docs.exists(),
        "has_manifest_latest": latest_manifest.exists(),
        "has_faiss_version": (v_faiss.exists() if v_faiss else None),
        "has_docs_version": (v_docs.exists() if v_docs else None),
        "has_manifest_version": (v_manifest.exists() if v_manifest else None),
        "manifest_path": str(mf),
    }
