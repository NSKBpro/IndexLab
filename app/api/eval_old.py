from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import pandas as pd, io, json
from typing import List, Dict, Any, Tuple, Optional
from ..core.config import settings
from ..ingest.indexer import load_index, search
from ..ingest.embedder import embed_texts
from sklearn.metrics import ndcg_score

router = APIRouter()

def _load_gold(upload: UploadFile) -> pd.DataFrame:
    name = upload.filename.lower()
    data = upload.file.read()  # bytes
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(data))
    elif name.endswith((".xlsx", ".xls", ".xlsm")):
        df = pd.read_excel(io.BytesIO(data))
    elif name.endswith(".json"):
        df = pd.read_json(io.BytesIO(data))
    else:
        raise HTTPException(400, "Provide CSV/XLSX/JSON with columns: question,expected_id")
    if not {"question","expected_id"}.issubset(df.columns):
        raise HTTPException(400, "Missing required columns: question, expected_id")
    # Normalize to str
    df["question"] = df["question"].astype(str)
    df["expected_id"] = df["expected_id"].astype(str)
    return df

def _open_index(index_name: str):
    p = settings.indexes_dir / f"{index_name}.faiss"
    if not p.exists(): raise HTTPException(404, f"index not found: {index_name}")
    backend, ids, manifest = load_index(p)
    docs = json.loads(p.with_suffix(".docs.json").read_text(encoding="utf-8"))
    return backend, ids, manifest, docs

def _eval_once(index_name: str, df: pd.DataFrame, k: int, include_hits: int = 0):
    backend, ids, manifest, docs = _open_index(index_name)
    id_pos = {i: p for p, i in enumerate(ids)}

    hits_total = 0
    rr_sum = 0.0
    ndcg_y_true, ndcg_y_score = [], []
    results: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        q = row["question"]
        exp = row["expected_id"]
        q_emb = embed_texts([q], manifest["model"], manifest.get("normalize", True))
        print(q_emb)
        res = search(backend, q_emb, max(k, 10), ids)  # [(id, score)]
        print(res)
        id_list = [i for (i, _) in res][:k]
        score_list = [float(s) for (_, s) in res][:k]

        found = exp in id_list
        rank = (id_list.index(exp) + 1) if found else None
        if found:
            hits_total += 1
            rr_sum += 1.0 / rank

        y_true = [1.0 if i == exp else 0.0 for i in id_list]
        y_score= score_list
        print(y_score)
        ndcg_y_true.append(y_true if y_true else [0.0])
        ndcg_y_score.append(y_score if y_score else [0.0])
        print(include_hits)
        if include_hits and include_hits > 0:
            show = min(include_hits, len(id_list))
            hits_view = [
                {"id": id_list[j], "score": score_list[j],
                 "preview": (docs[id_list[j]][:180] + "…") if len(docs[id_list[j]]) > 180 else docs[id_list[j]]}
                for j in range(show)
            ]
        else:
            hits_view = []

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

@router.post("/eval")
async def eval_api(
    index_name: str = Form(...),
    k: int = Form(5),
    file: UploadFile = File(...),
    return_details: bool = Form(False),
    include_hits: int = Form(0)
):
    df = _load_gold(file)
    out = _eval_once(index_name, df, k, include_hits if return_details else 0)
    if not return_details:
        # strip bulky per-question results unless requested
        out.pop("results", None)
    return out

@router.post("/eval_compare")
async def eval_compare_api(
    left_index: str = Form(...),
    right_index: str = Form(...),
    k: int = Form(5),
    file: UploadFile = File(...),
    include_hits: int = Form(0)
):
    """
    Compare two indexes on the same gold set; return per-question ranks and deltas.
    """
    df = _load_gold(file)
    left = _eval_once(left_index, df, k, include_hits)
    right = _eval_once(right_index, df, k, include_hits)

    # join on question+expected_id in order
    regressions = []
    improvements = []
    combined: List[Dict[str, Any]] = []
    for a, b in zip(left["results"], right["results"]):
        # They align row-for-row because both used the same df order
        a_rank, b_rank = a.get("rank"), b.get("rank")
        delta = None
        if a_rank is not None and b_rank is not None:
            delta = b_rank - a_rank  # positive = worse on right, negative = better
        elif a_rank is None and b_rank is not None:
            delta = -999  # recovered (was miss -> now hit)
        elif a_rank is not None and b_rank is None:
            delta = 999   # regression (was hit -> now miss)

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
            row["left_hits"] = a["hits"]
            row["right_hits"] = b["hits"]

        combined.append(row)
        if delta is not None:
            if delta > 0:
                regressions.append(row)
            elif delta < 0:
                improvements.append(row)

    summary = {
        "k": k,
        "total": left["total"],
        "left": {k:v for k,v in left.items() if k != "results"},
        "right": {k:v for k,v in right.items() if k != "results"},
        "regressions_count": len([r for r in combined if r["delta"] is not None and r["delta"] > 0]),
        "improvements_count": len([r for r in combined if r["delta"] is not None and r["delta"] < 0]),
        "changed_count": len([r for r in combined if (r["delta"] is not None and abs(r["delta"]) not in (0,999)) or (r["delta"] in (999, -999))]),
        "results": combined
    }
    return summary
