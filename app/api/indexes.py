# app/api/indexes_api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
from ..core.config import settings

router = APIRouter()

def _latest_version_info(index_dir: Path) -> tuple[str|None, str|None]:
    vdir = index_dir / "versions"
    if not vdir.exists():
        return None, None
    # prefer folder meta.json
    folder_metas = []
    for d in vdir.glob("*"):
        if d.is_dir() and (d / "meta.json").exists():
            try:
                data = json.loads((d / "meta.json").read_text(encoding="utf-8"))
                folder_metas.append((data.get("version"), data.get("created_at")))
            except Exception:
                pass
    if folder_metas:
        folder_metas.sort(key=lambda x: (x[1] or "", x[0] or ""), reverse=True)
        return folder_metas[0]
    # fallback to flat files
    items = sorted(vdir.glob("*.json"), reverse=True)
    if not items:
        return None, None
    data = json.loads(items[0].read_text(encoding="utf-8"))
    return data.get("version"), data.get("created_at")


@router.get("/indexes")
def list_indexes():
    root = settings.indexes_dir
    if not root.exists():
        return {"indexes": []}

    names = set()

    # 1) infer from *.faiss files
    for f in root.glob("*.faiss"):
        names.add(f.stem)

    # 2) infer from version folders: <root>/<name>/versions/*.json
    for d in root.iterdir():
        if d.is_dir() and (d / "versions").exists():
            names.add(d.name)

    # materialize with meta
    items = []
    for name in sorted(names):
        index_dir = root / name
        latest_version, created_at = _latest_version_info(index_dir)
        items.append({
            "index_name": name,
            "latest_version": latest_version,
            "created_at": created_at,
            "has_versions": bool(latest_version),
        })
    return {"indexes": items}
