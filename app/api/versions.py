# app/api/versioning_api.py
from __future__ import annotations
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from ..core.config import settings

# Mounted as /api/versioning via main.py include
router = APIRouter(prefix="/versioning", tags=["versioning"])

def _root_for(index_name: str) -> Path:
    return settings.indexes_dir / index_name / "versions"

def _read_json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

@router.get("/_ping")
def ping():
    return {"ok": True, "indexes_dir": str(settings.indexes_dir)}

@router.get("/{index_name}")
def list_index_versions(index_name: str):
    vroot = _root_for(index_name)
    versions: list[dict] = []
    if not vroot.exists():
        return {"index_name": index_name, "versions": versions}

    # New style: directories with meta.json
    for d in sorted(vroot.glob("*"), reverse=True):
        if d.is_dir():
            meta = _read_json(d / "meta.json")
            if meta and meta.get("version"):
                versions.append(meta)

    # Legacy flat files: <versions>/<timestamp>.json
    for f in sorted(vroot.glob("*.json"), reverse=True):
        if f.name == "meta.json":
            continue
        # Skip ones that we already have from folder meta (avoid dupes)
        if f.stem in {v.get("version") for v in versions if v.get("version")}:
            continue
        data = _read_json(f)
        if data and data.get("version"):
            versions.append(data)

    # Sort by created_at (desc) if present, else by version desc
    versions.sort(key=lambda v: (v.get("created_at") or "", v.get("version") or ""), reverse=True)
    return {"index_name": index_name, "versions": versions}

@router.get("/{index_name}/{version}")
def get_index_version(index_name: str, version: str):
    vroot = _root_for(index_name)
    # Try new style
    f = vroot / version / "meta.json"
    if f.exists():
        data = _read_json(f)
        if data:
            return data
        raise HTTPException(500, "Invalid meta.json")
    # Fallback legacy
    f2 = vroot / f"{version}.json"
    if f2.exists():
        data = _read_json(f2)
        if data:
            return data
        raise HTTPException(500, "Invalid version file")
    raise HTTPException(404, "Version not found")

@router.get("/{index_name}/{version}/artifacts")
def get_version_artifacts(index_name: str, version: str):
    """Returns paths (server-local) to artifacts for debugging/admin."""
    vdir = _root_for(index_name) / version
    return {
        "index_name": index_name,
        "version": version,
        "exists": vdir.exists(),
        "paths": {
            "faiss": str(vdir / f"{index_name}.faiss"),
            "docs": str(vdir / f"{index_name}.docs.json"),
            "manifest": str(vdir / "manifest.json"),
            "meta": str(vdir / "meta.json"),
        }
    }
