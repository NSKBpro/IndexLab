# app/api/eval_api.py
from __future__ import annotations

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import io, json

from ..core.config import settings
from ..ingest.indexer import load_index, search
from ..ingest.embedder import embed_texts
from sklearn.metrics import ndcg_score
import numpy as np

router = APIRouter()

# --------------------------
# Helpers
# --------------------------

def _load_gold(upload: UploadFile) -> pd.DataFrame:
    """Load CSV/XLSX/JSON with columns: question, expected_id (case-insensitive)."""
    if upload is None:
        raise HTTPException(400, "Missing dataset file")
    name = (upload.filename or "").lower()
    data = upload.file.read()  # bytes

    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
    elif name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(io.BytesIO(data))
    elif name.endswith(".json"):
        df = pd.read_json(io.BytesIO(data))
    else:
        raise HTTPException(400, "Provide CSV/XLSX/JSON with columns: question, expected_id")

    # Normalize headers (case-insensitive)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"question", "expected_id"}.issubset(df.columns):
        raise HTTPException(400, "Missing required columns: question, expected_id")

    # Normalize values to strings (trim whitespace)
    df["question"] = df["question"].astype(str).str.strip()
    df["expected_id"] = df["expected_id"].astype(str).str.strip()
    # Drop rows with empty values
    df = df[(df["question"] != "") & (df["expected_id"] != "")]
    return df.reset_index(drop=True)


def _resolve_index_path(index_name: str, version: Optional[str] = None) -> Path:
    """
    Resolve an index name (and optional version) to a .faiss path.

      - version=None:
          <indexes_dir>/<name>.faiss
      - version="YYYYMMDD-HHMMSS":
          <indexes_dir>/<name>/versions/<version>/<name>.faiss

      If index_name is an absolute path to a .faiss file, use as-is.
    """
    p = Path(index_name)
    if p.is_absolute() and p.suffix == ".faiss":
        return p

    base = Path(settings.indexes_dir)
    if version:
        vp = base / index_name / "versions" / version / f"{index_name}.faiss"
        return vp

    return base / (p.name if p.suffix == ".faiss" else f"{p.name}.faiss")


def _safe_read_json(path: Path) -> Optional[Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return None


def _open_index(index_name: str, version: Optional[str] = None):
    """
    Open index (optionally a specific version) and normalize return shapes into:
      backend, ids(list[str]), manifest(dict), docs(any or None)
    """
    p = _resolve_index_path(index_name, version)
    if not p.exists():
        where = f"{index_name}@{version}" if version else index_name
        raise HTTPException(404, f"index not found: {where}")

    # Try loader with Path, name, string path (be tolerant)
    tried = []
    res = None
    for arg in (p, p.stem, str(p)):
        try:
            res = load_index(arg)
            tried.append(f"load_index({arg!r}) -> {type(res).__name__}")
            if res:
                break
        except Exception as e:
            tried.append(f"load_index({arg!r}) EXC: {e}")

    if not res:
        raise HTTPException(500, detail="load_index(...) returned no result.\n" + "\n".join(tried))

    backend: Any = None
    ids: Optional[List[str]] = None
    manifest: Dict[str, Any] = {}
    docs: Any = None

    # Normalize common shapes
    if isinstance(res, tuple):
        if len(res) == 3 and isinstance(res[1], (list, tuple)):
            backend, ids_raw, manifest = res
            ids = [str(x) for x in ids_raw]
            manifest = dict(manifest or {})
        elif len(res) >= 4:
            backend, manifest, ids_raw, docs = res[0], res[1], res[2], res[3]
            ids = [str(x) for x in ids_raw]
            manifest = dict(manifest or {})
        else:
            b, a, c = (res + (None, None, None))[:3]
            if b is not None: backend = b
            if isinstance(a, (list, tuple)): ids = [str(x) for x in a]
            if isinstance(c, dict): manifest = dict(c or {})
    elif isinstance(res, dict):
        backend = res.get("backend")
        if isinstance(res.get("ids"), (list, tuple)):
            ids = [str(x) for x in res["ids"]]
        elif isinstance(res.get("id_map"), dict):
            vals = list(res["id_map"].values())
            if vals and isinstance(vals[0], (str, int)):
                ids = [str(x) for x in vals]
        manifest = dict(res.get("manifest") or res.get("meta") or {})
        docs = res.get("docs")
    else:
        backend = getattr(res, "backend", None)
        ids_attr = getattr(res, "ids", None)
        if isinstance(ids_attr, (list, tuple)):
            ids = [str(x) for x in ids_attr]
        manifest = dict(getattr(res, "manifest", {}) or getattr(res, "meta", {}) or {})
        docs = getattr(res, "docs", None)

    # Fill from disk if needed (version-aware)
    if not manifest:
        mf = p.with_suffix(".manifest.json")
        mfd = _safe_read_json(mf)
        if isinstance(mfd, dict):
            manifest = mfd
    if docs is None:
        docs = _safe_read_json(p.with_suffix(".docs.json"))

    if backend is None or not ids:
        raise HTTPException(500, detail="load_index(...) did not provide backend + ids.\n" + "\n".join(tried))

    return backend, ids, manifest, docs


def _embed_one(question: str, manifest: Dict[str, Any]):
    """Embed a single question with tolerant signature handling."""
    model = manifest["model"]
    normalize = manifest.get("normalize", True)
    try:
        vecs = embed_texts([question], model, normalize)
    except TypeError:
        vecs = embed_texts([question])

    if not isinstance(vecs, np.ndarray) or vecs.size == 0:
        raise HTTPException(500, "embed_texts returned no vectors or an invalid type")

    return vecs


def _preview_from_docs(docs: Any, doc_id: str, limit: int = 180) -> str:
    """
    Create a safe preview string given docs collection which can be:
      - dict: {doc_id: "text"} or {doc_id: {...}}
      - list: [ "text", ... ] or [ {"id": "...", "text": "..."}, ... ]
    """
    try:
        if isinstance(docs, dict):
            v = docs.get(doc_id)
            if v is None:
                return ""
            if isinstance(v, dict):
                txt = v.get("text") or v.get("content") or v.get("body") or ""
            else:
                txt = str(v)
            txt = str(txt)
            return (txt[:limit] + "…") if len(txt) > limit else txt

        if isinstance(docs, list):
            if doc_id.isdigit():
                idx = int(doc_id)
                if 0 <= idx < len(docs):
                    v = docs[idx]
                    txt = v.get("text") if isinstance(v, dict) else str(v)
                    txt = str(txt or "")
                    return (txt[:limit] + "…") if len(txt) > limit else txt
            for v in docs:
                if isinstance(v, dict) and str(v.get("id")) == doc_id:
                    txt = str(v.get("text") or v.get("content") or v.get("body") or "")
                    return (txt[:limit] + "…") if len(txt) > limit else txt
    except Exception:
        pass
    return ""


def _eval_once(index_name: str, df: pd.DataFrame, k: int, include_hits: int = 0, version: Optional[str] = None):
    backend, ids, manifest, docs = _open_index(index_name, version=version)
    ids = [str(x) for x in ids]  # ensure string ids everywhere

    hits_total = 0
    rr_sum = 0.0
    ndcg_y_true, ndcg_y_score = [], []
    results: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        q = row["question"]
        exp = str(row["expected_id"])

        q_emb = _embed_one(q, manifest)
        # your search returns [(id, score)] — keep a little cushion max(k, 10)
        res = search(backend, q_emb, max(k, 10), ids)  # -> [(id, score)]
        id_list = [str(i) for (i, _) in res][:k]
        score_list = [float(s) for (_, s) in res][:k]

        found = exp in id_list
        rank = (id_list.index(exp) + 1) if found else None
        if found:
            hits_total += 1
            rr_sum += 1.0 / rank

        # NDCG vectors (binary relevance for expected_id)
        y_true = [1.0 if i == exp else 0.0 for i in id_list] or [0.0]
        y_score = score_list or [0.0]
        ndcg_y_true.append(y_true)
        ndcg_y_score.append(y_score)

        # Optional top hits preview
        hits_view = []
        if include_hits and include_hits > 0:
            show = min(include_hits, len(id_list))
            for j in range(show):
                _id = id_list[j]
                _score = score_list[j]
                preview = _preview_from_docs(docs, _id, 180) if docs is not None else ""
                hits_view.append({"id": _id, "score": _score, "preview": preview})

        results.append({
            "question": q,
            "expected_id": exp,
            "found": bool(found),
            "rank": rank,
            "top_ids": id_list,
            "top_scores": score_list,
            "hits": hits_view
        })

    total = len(df)
    recall_at_k = hits_total / total if total else 0.0
    mrr = rr_sum / total if total else 0.0
    ndcg = float(ndcg_score(ndcg_y_true, ndcg_y_score)) if ndcg_y_true else 0.0

    return {
        "k": k, "total": total,
        "recall_at_k": recall_at_k, "mrr": mrr, "ndcg": ndcg,
        "results": results
    }

# --------------------------
# Routes (FORM-DATA)
# --------------------------

@router.post("/eval")
async def eval_api(
    index_name: str = Form(...),
    k: int = Form(5),
    file: UploadFile = File(...),
    return_details: bool = Form(False),
    include_hits: int = Form(0),
    index_version: Optional[str] = Form(None),
):
    """
    Evaluate a single index (optionally a specific version).
    Upload CSV/XLSX/JSON with columns: question, expected_id.
    """
    df = _load_gold(file)
    out = _eval_once(index_name, df, k, include_hits if return_details else 0, version=index_version)
    if not return_details:
        out.pop("results", None)
    return out


@router.post("/eval_compare")
async def eval_compare_api(
    left_index: str = Form(...),
    right_index: str = Form(...),
    k: int = Form(5),

    # Accept EITHER a single "file" or separate sides
    file: Optional[UploadFile] = File(default=None),
    file_left: Optional[UploadFile] = File(default=None),
    file_right: Optional[UploadFile] = File(default=None),

    # Optional extra fields (currently unused in core logic, but accepted)
    include_hits: int = Form(0),
    left_version: Optional[str] = Form(None),
    right_version: Optional[str] = Form(None),
    left_hybrid: Optional[str] = Form(None),
    right_hybrid: Optional[str] = Form(None),
):
    """
    Compare two indexes using the same gold set or separate gold sets.
    Frontend may send:
      - file                        (use for both)
      - file_left [and optionally file_right]
    Versions can be passed via left_version / right_version.
    """
    # Normalize files
    f_left = file_left or file
    f_right = file_right or f_left or file
    if f_left is None:
        raise HTTPException(400, "Field required: file or file_left")

    df_left = _load_gold(f_left)
    df_right = _load_gold(f_right) if (f_right is not None) else df_left

    # For now we assume the same row order; if lengths differ, we trim to min length
    if len(df_left) != len(df_right):
        min_len = min(len(df_left), len(df_right))
        df_left = df_left.iloc[:min_len].reset_index(drop=True)
        df_right = df_right.iloc[:min_len].reset_index(drop=True)

    left = _eval_once(left_index, df_left, k, include_hits, version=left_version)
    right = _eval_once(right_index, df_right, k, include_hits, version=right_version)

    combined: List[Dict[str, Any]] = []
    for a, b in zip(left.get("results", []), right.get("results", [])):
        a_rank, b_rank = a.get("rank"), b.get("rank")
        delta = None
        if a_rank is not None and b_rank is not None:
            delta = b_rank - a_rank            # positive = worse on right, negative = better
        elif a_rank is None and b_rank is not None:
            delta = -999                       # recovered (miss -> hit)
        elif a_rank is not None and b_rank is None:
            delta = 999                        # regression (hit -> miss)

        row = {
            "question": a["question"],
            "expected_id": a["expected_id"],
            "left_rank": a_rank,
            "right_rank": b_rank,
            "delta": delta,
            "left_found": a["found"],
            "right_found": b["found"],
        }
        if include_hits:
            row["left_hits"] = a.get("hits", [])
            row["right_hits"] = b.get("hits", [])
        combined.append(row)

    summary = {
        "k": k,
        "total": left["total"],
        "left": {kk: vv for kk, vv in left.items() if kk != "results"},
        "right": {kk: vv for kk, vv in right.items() if kk != "results"},
        "regressions_count": sum(1 for r in combined if r["delta"] is not None and r["delta"] > 0),
        "improvements_count": sum(1 for r in combined if r["delta"] is not None and r["delta"] < 0),
        "changed_count": sum(1 for r in combined if (r["delta"] in (999, -999)) or (r["delta"] not in (None, 0)) ),
        "results": combined,
    }
    return summary
