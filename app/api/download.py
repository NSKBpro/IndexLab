# app/api/download_api.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pathlib import Path
import io, zipfile

from ..core.config import settings

router = APIRouter(tags=["download"])

def _version_dir(index_name: str, version: str) -> Path | None:
    p = settings.indexes_dir / index_name / "versions" / version
    return p if p.exists() and p.is_dir() else None

def _latest_files(index_name: str) -> list[Path]:
    base = settings.indexes_dir
    stems = [
        f"{index_name}.faiss",
        f"{index_name}.docs.json",
        f"{index_name}.manifest.json",
        f"{index_name}.ids.json",  # optional/legacy if you have it
    ]
    return [base / s for s in stems if (base / s).exists()]

@router.get("/download/{index_name}")
def download_index(index_name: str, version: str | None = Query(default=None)):
    # tiny sanitize
    if "/" in index_name or "\\" in index_name:
        raise HTTPException(400, "invalid index name")

    if version:
        vdir = _version_dir(index_name, version)
        if not vdir:
            raise HTTPException(404, f"version not found: {index_name} v{version}")
        files = sorted([p for p in vdir.iterdir() if p.is_file()])
        if not files:
            raise HTTPException(404, "no files in version folder")
        arc_prefix = f"{index_name}_v{version}/"
        fname = f"{index_name}_v{version}.zip"
    else:
        files = _latest_files(index_name)
        if not files:
            raise HTTPException(404, f"no latest artifacts for {index_name}")
        arc_prefix = f"{index_name}_latest/"
        fname = f"{index_name}_latest.zip"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in files:
            zf.write(fp, arcname=arc_prefix + fp.name)
    buf.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{fname}"'}
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
