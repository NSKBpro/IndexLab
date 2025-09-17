# app/api/stats_api.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, HTTPException, Query

from ..core.config import settings

router = APIRouter(tags=["analytics"])


# ---------- path helpers ----------

def _manifest_path_latest(index_name: str) -> Path | None:
    """
    Latest manifest lives next to the latest .faiss:
      <indexes_dir>/<index_name>.manifest.json  (or .faiss.manifest.json)
    """
    base = settings.indexes_dir / f"{index_name}.faiss"
    for cand in [base.with_suffix(".manifest.json"), base.with_suffix(".faiss.manifest.json")]:
        if cand.exists():
            return cand
    return None


def _docs_path_latest(index_name: str) -> Path | None:
    """
    Latest docs next to latest .faiss:
      <indexes_dir>/<index_name>.docs.json
    """
    p = (settings.indexes_dir / f"{index_name}.faiss").with_suffix(".docs.json")
    return p if p.exists() else None


def _ids_path_latest(index_name: str) -> Path | None:
    """
    Optional ids file:
      <indexes_dir>/<index_name>.ids.json
    """
    p = (settings.indexes_dir / f"{index_name}.faiss").with_suffix(".ids.json")
    return p if p.exists() else None


def _manifest_path_version(index_name: str, version: str) -> Path | None:
    """
    Versioned manifest:
      <indexes_dir>/<index_name>/versions/<version>/manifest.json
    """
    p = settings.indexes_dir / index_name / "versions" / version / "manifest.json"
    return p if p.exists() else None


def _docs_path_version(index_name: str, version: str) -> Path | None:
    """
    Versioned docs:
      <indexes_dir>/<index_name>/versions/<version>/<index_name>.docs.json
    """
    p = settings.indexes_dir / index_name / "versions" / version / f"{index_name}.docs.json"
    return p if p.exists() else None


def _ids_path_version(index_name: str, version: str) -> Path | None:
    """
    Versioned ids:
      <indexes_dir>/<index_name>/versions/<version>/<index_name>.ids.json
    """
    p = settings.indexes_dir / index_name / "versions" / version / f"{index_name}.ids.json"
    return p if p.exists() else None


# ---------- IO helpers ----------

def _read_json(path: Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Invalid JSON at {path}: {e}")


def _lengths_from_docs(docs: Dict[str, Any], ids: Optional[List[str]]) -> List[int]:
    """
    Compute chunk lengths.
    - docs can be {id: "text"} OR {id: {"text": "..."}}
    - if ids provided, preserve that order; else use docs.keys() order
    """
    order = ids if ids else list(docs.keys())
    lens: List[int] = []
    for i in order:
        v = docs.get(i)
        if isinstance(v, str):
            lens.append(len(v))
        elif isinstance(v, dict):
            t = v.get("text")
            if isinstance(t, str):
                lens.append(len(t))
        # else: missing/invalid -> skip
    return lens


def _percentile(vals: List[int], p: float) -> Optional[float]:
    if not vals:
        return None
    a = sorted(vals)
    k = (len(a) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(a[int(k)])
    return a[f] + (a[c] - a[f]) * (k - f)


def _histogram(vals: List[int], bins: int = 20) -> Dict[str, Any]:
    if not vals:
        return {"bins": [], "counts": []}
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        # Single spike case
        return {"bins": [vmin, vmax], "counts": [len(vals)]}
    step = max(1, math.ceil((vmax - vmin) / bins))
    edges = [vmin + i * step for i in range(bins)]
    edges.append(vmax)
    counts = [0] * (len(edges) - 1)
    for x in vals:
        if x == vmax:
            idx = len(counts) - 1
        else:
            idx = min(len(counts) - 1, (x - vmin) // step)
        counts[idx] += 1
    return {"bins": edges, "counts": counts}


# ---------- route ----------

@router.get("/stats/{index_name}")
def index_stats(
    index_name: str,
    version: Optional[str] = Query(default=None, description="Version id (e.g. 20250917-091032). Omit for latest."),
):
    """
    Analytics for latest or specific version:
      - chunks/count (from docs + ids if present)
      - length stats (avg, p95, max, min)
      - histogram {bins, counts}
      - manifest (object) + flattened build fields
    """
    # Resolve paths
    if version:
        manifest_path = _manifest_path_version(index_name, version)
        if not manifest_path:
            raise HTTPException(404, f"manifest not found for '{index_name}' version '{version}'")
        docs_path = _docs_path_version(index_name, version)
        ids_path = _ids_path_version(index_name, version)
    else:
        base_faiss = settings.indexes_dir / f"{index_name}.faiss"
        if not base_faiss.exists():
            raise HTTPException(404, "index not found")
        manifest_path = _manifest_path_latest(index_name)
        if not manifest_path:
            raise HTTPException(404, f"manifest not found for '{index_name}' (latest)")
        docs_path = _docs_path_latest(index_name)
        ids_path = _ids_path_latest(index_name)

    manifest = _read_json(manifest_path)
    docs = _read_json(docs_path)
    ids_list = None
    if ids_path:
        try:
            raw_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            # ids file can be an array of strings, or {ids: [...]}
            if isinstance(raw_ids, list):
                ids_list = [str(x) for x in raw_ids]
            elif isinstance(raw_ids, dict) and isinstance(raw_ids.get("ids"), list):
                ids_list = [str(x) for x in raw_ids["ids"]]
        except Exception:
            ids_list = None  # non-fatal

    lengths = _lengths_from_docs(docs, ids_list)
    count = len(lengths)
    len_min = min(lengths) if lengths else 0
    len_max = max(lengths) if lengths else 0
    len_avg = (sum(lengths) / count) if count else 0.0
    len_p95 = _percentile(lengths, 0.95) or 0.0
    hist = _histogram(lengths, bins=20) if lengths else {"bins": [], "counts": []}

    return {
        "index_name": index_name,
        "version": version,
        # counts (UI accepts either 'chunks' or 'count')
        "chunks": count,
        "count": count,
        # length stats
        "len_min": len_min,
        "len_max": len_max,
        "len_avg": float(len_avg),
        "len_p95": float(len_p95),
        # histogram (UI normalizer handles {hist:{bins,counts}})
        "hist": hist,
        # keep full manifest (UI uses this to render Build config + Raw JSON)
        "manifest": manifest,
        # flattened convenience fields for cards
        "model": manifest.get("model"),
        "dim": manifest.get("dim"),
        "backend": manifest.get("backend"),
        "normalize": manifest.get("normalize"),
        "chunking": manifest.get("chunking", {}),
        "created_at": manifest.get("created_at"),
    }
