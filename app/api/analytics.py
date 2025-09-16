from fastapi import APIRouter, HTTPException
import json, numpy as np
from ..core.config import settings

router = APIRouter()

@router.get("/stats/{index_name}")
def index_stats(index_name: str):
    p = settings.indexes_dir / f"{index_name}.faiss"
    if not p.exists(): raise HTTPException(404, "index not found")
    manifest = json.loads(p.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    ids = json.loads(p.with_suffix(".ids.json").read_text(encoding="utf-8"))
    docs = json.loads(p.with_suffix(".docs.json").read_text(encoding="utf-8"))
    lengths = [len(docs[i]) for i in ids]
    import numpy as np
    return {
        "manifest": manifest,
        "chunks": len(ids),
        "len_avg": float(np.mean(lengths)) if lengths else 0,
        "len_p95": float(np.percentile(lengths, 95)) if lengths else 0,
        "len_max": max(lengths) if lengths else 0
    }
