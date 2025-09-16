# ruff: noqa: D100,D101,D102,D103
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..core.config import settings
from ..ingest.indexer import load_index, search
from ..ingest.embedder import embed_texts
from ..ingest.hybrid import bm25_build, bm25_search, rrf_fuse

router = APIRouter()

# ======================
# Models
# ======================

class SearchReq(BaseModel):
    index_name: str
    query: str
    k: Optional[int] = None
    hybrid: bool = False
    bm25_k: int = 50

class CompareReq(BaseModel):
    left_index: str
    right_index: str
    query: str
    k: int = 5

# ======================
# Helpers
# ======================

_DOC_KEY = re.compile(r"^#?(\d+)#(\d+)$")  # "#223#0" or "223#0"

def _parse_doc_key(doc_id: str) -> tuple[int | None, int | None]:
    m = _DOC_KEY.match(doc_id.strip())
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _load_optional_metadata(idx_path) -> Dict[str, Dict[str, Any]]:
    """
    Optional per-doc_id metadata next to .faiss:
      <index>.meta.json -> { "<doc_id>": { "title": "...", "path": "...", "url": "...", "page": 1, "section": "..." } }
    """
    p = idx_path.with_suffix(".meta.json")
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _load_optional_sources(idx_path) -> Dict[str, Any]:
    """
    Optional provenance sidecar next to .faiss:
      <index>.sources.json -> { "docs": { "1": {...}, "2": {...} }, "chunking": {...} }
    """
    p = idx_path.with_suffix(".sources.json")
    try:
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _build_source(
    doc_id: str,
    internal_idx: int,
    index_name: str,
    meta_map: Dict[str, Dict[str, Any]],
    sources_map: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compose a 'source' object (no mutation of your text).
    Always includes: index, doc_id, internal_idx, parsed doc/chunk (if #x#y).
    Enriches with meta_map (per doc_id) + sources_map (per doc number).
    """
    meta = meta_map.get(doc_id) or {}
    dno, cno = _parse_doc_key(doc_id)

    out: Dict[str, Any] = {
        "index": index_name,
        "doc_id": doc_id,
        "internal_idx": internal_idx,
    }
    if dno is not None and cno is not None:
        out["parsed"] = {"doc": dno, "chunk": cno}

    # per-doc_id metadata (path/url/title/page/section/etc)
    for key in ("title", "path", "url", "page", "section", "chunk_id", "created_at", "hash"):
        if key in meta:
            out[key] = meta[key]

    # provenance (per doc number) from <index>.sources.json
    docs_prov = (sources_map.get("docs") or {})
    if dno is not None:
        prov = docs_prov.get(str(dno)) or docs_prov.get(dno)
        if prov:
            for key in (
                "origin_type", "origin_path", "origin_url", "collection",
                "content_type", "bytes", "sha256", "ingested_at", "container"
            ):
                if key in prov:
                    out[key] = prov[key]

    # chunking strategy (index-level)
    if "chunking" in sources_map:
        out["chunking"] = sources_map["chunking"]

    return out

def _normalize_hits_to_indices(
    hits: List[Tuple[object, Optional[float]]],
    ids: List[str],
) -> List[Tuple[int, Optional[float]]]:
    """
    Accept hits as [(int_idx, score)] or [(doc_id:str, score)] and return [(int_idx, score)].
    Unknown doc_ids are skipped.
    """
    id_to_pos = {doc_id: pos for pos, doc_id in enumerate(ids)}
    norm: List[Tuple[int, Optional[float]]] = []
    for key, score in hits:
        if isinstance(key, int):
            if 0 <= key < len(ids):
                norm.append((key, score))
        elif isinstance(key, str):
            pos = id_to_pos.get(key)
            if pos is not None:
                norm.append((pos, score))
    return norm

# ----- highlighting (regex-based, no text mutation) -----

def _tokenize_query(q: str) -> List[str]:
    return [t for t in re.split(r"\W+", q) if t]

def _find_spans(text: str, terms: List[str], max_hits_per_term: int = 8) -> Dict[str, List[Tuple[int,int]]]:
    spans: Dict[str, List[Tuple[int,int]]] = {}
    if not text or not terms:
        return spans
    for t in terms:
        if not t:
            continue
        pat = re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
        hits: List[Tuple[int,int]] = []
        for m in pat.finditer(text):
            hits.append((m.start(), m.end()))
            if len(hits) >= max_hits_per_term:
                break
        if hits:
            spans[t] = hits
    return spans

def _slice_preview(text: str, maxlen: int = 220) -> str:
    return (text[:maxlen] + "…") if len(text) > maxlen else text

# ======================
# Routes
# ======================

@router.get("/chunk")
def get_chunk(
    index_name: str = Query(..., description="Index name (without .faiss)"),
    doc_id: str = Query(..., description="Exact doc_id returned in hits (e.g. '#223#0' or path-like id)"),
    q: str | None = Query(None, description="Optional query string for highlight spans"),
):
    """
    Return the FULL, RAW chunk text for a given doc_id + index, plus source metadata and optional highlight spans.
    """
    idx_path = settings.indexes_dir / f"{index_name}.faiss"
    if not idx_path.exists():
        raise HTTPException(404, "index not found")

    backend, ids, manifest = load_index(idx_path)  # backend unused here
    docs_map = json.loads(idx_path.with_suffix(".docs.json").read_text(encoding="utf-8"))
    meta_map = _load_optional_metadata(idx_path)
    sources_map = _load_optional_sources(idx_path)

    # allow numeric doc_id to mean internal index
    if doc_id not in docs_map:
        try:
            maybe_pos = int(doc_id)
            if 0 <= maybe_pos < len(ids):
                real_id = ids[maybe_pos]
                text = docs_map.get(real_id, "")
                terms = _tokenize_query(q or "")
                return {
                    "doc_id": real_id,
                    "text": text,
                    "highlights": _find_spans(text, terms) if terms else {},
                    "source": _build_source(real_id, maybe_pos, index_name, meta_map, sources_map),
                    "manifest": manifest,
                }
        except Exception:
            pass
        raise HTTPException(404, f"doc_id '{doc_id}' not found in docs map")

    # find internal index
    try:
        internal_idx = ids.index(doc_id)
    except ValueError:
        internal_idx = -1

    text = docs_map[doc_id]
    terms = _tokenize_query(q or "")
    return {
        "doc_id": doc_id,
        "text": text,
        "highlights": _find_spans(text, terms) if terms else {},
        "source": _build_source(doc_id, internal_idx, index_name, meta_map, sources_map),
        "manifest": manifest,
    }

@router.post("/search")
def search_api(req: SearchReq):
    idx_path = settings.indexes_dir / f"{req.index_name}.faiss"
    if not idx_path.exists():
        raise HTTPException(404, "index not found")

    backend, ids, manifest = load_index(idx_path)
    docs_map = json.loads(idx_path.with_suffix(".docs.json").read_text(encoding="utf-8"))
    texts = [docs_map[i] for i in ids]
    meta_map = _load_optional_metadata(idx_path)
    sources_map = _load_optional_sources(idx_path)

    q_emb = embed_texts([req.query], manifest["model"], manifest.get("normalize", True))
    k = req.k or settings.top_k
    terms = _tokenize_query(req.query)

    # search() may return [(int_idx, score)] or [(doc_id, score)]
    raw_vec_hits = search(backend, q_emb, max(k, 50), ids)
    vec_hits = _normalize_hits_to_indices(raw_vec_hits, ids)

    if req.hybrid:
        bm = bm25_build(texts)
        bm_hits_idx = bm25_search(bm, req.query, req.bm25_k)  # internal indices
        vec_hits_pos = [(pos, s) for (pos, s) in vec_hits]
        fused = rrf_fuse(vec_hits_pos, bm_hits_idx, k)        # -> [(internal_idx, _)]
        chosen = [(i, None) for (i, _ignored) in fused]       # do not fabricate scores
    else:
        chosen = vec_hits[:k]

    vec_score_map = {i: s for (i, s) in vec_hits if s is not None}

    hits = []
    for (internal_idx, s) in chosen[:k]:
        doc_id = ids[internal_idx]
        raw_text = docs_map.get(doc_id, "")
        preview = _slice_preview(raw_text, 220)   # unchanged content
        hl = _find_spans(preview, terms)          # spans inside preview only

        item = {
            "id": doc_id,
            "preview": preview,
            "highlights_preview": hl,
            "source": _build_source(doc_id, internal_idx, req.index_name, meta_map, sources_map),
        }
        score_to_show = s if s is not None else vec_score_map.get(internal_idx)
        if score_to_show is not None:
            try:
                item["score"] = float(score_to_show)
            except Exception:
                pass
        hits.append(item)

    return {"hits": hits, "manifest": manifest}

@router.post("/compare")
def compare_api(req: CompareReq):
    def run(ix_name: str):
        p = settings.indexes_dir / f"{ix_name}.faiss"
        if not p.exists():
            raise HTTPException(404, f"index {ix_name} not found")
        backend, ids, manifest = load_index(p)
        docs = json.loads(p.with_suffix(".docs.json").read_text(encoding="utf-8"))
        meta_map = _load_optional_metadata(p)
        sources_map = _load_optional_sources(p)
        q_emb = embed_texts([req.query], manifest["model"], manifest.get("normalize", True))
        raw_hits = search(backend, q_emb, req.k, ids)
        hits = _normalize_hits_to_indices(raw_hits, ids)
        out = []
        terms = _tokenize_query(req.query)
        for (i, s) in hits:
            doc_id = ids[i]
            txt = docs.get(doc_id, "")
            preview = _slice_preview(txt, 200)
            hl = _find_spans(preview, terms)
            item = {
                "id": doc_id,
                "preview": preview,
                "highlights_preview": hl,
                "source": _build_source(doc_id, i, ix_name, meta_map, sources_map),
            }
            if s is not None:
                try:
                    item["score"] = float(s)
                except Exception:
                    pass
            out.append(item)
        return out, manifest

    left, lman = run(req.left_index)
    right, rman = run(req.right_index)
    left_ids, right_ids = {h["id"] for h in left}, {h["id"] for h in right}
    overlap = list(left_ids & right_ids)
    return {"left": {"hits": left, "manifest": lman},
            "right": {"hits": right, "manifest": rman},
            "overlap": overlap}
