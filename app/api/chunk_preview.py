from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List, Dict, Any
import os, tempfile
from pathlib import Path

from ..ingest.reader import read_any
from ..ingest.normalize import normalize_df
from ..ingest.chunker import chunk_fixed, chunk_sentences, chunk_by_headings

router = APIRouter()

# You can raise these if you want bigger previews
HARD_LIMIT_CHARS = 2_000_000  # 2 MB of text hard cap to avoid OOM on giant files

def _chunk_text(txt: str, mode: str, size: int, overlap: int) -> List[str]:
    if mode == "fixed_chars":
        return chunk_fixed(txt, size, overlap)
    if mode == "sentences":
        return chunk_sentences(txt, size, overlap)
    if mode == "headings":
        return chunk_by_headings(txt, size, overlap)
    return chunk_fixed(txt, size, overlap)

def _stats(chunks: List[str]) -> Dict[str, Any]:
    if not chunks:
        return {"count": 0, "avg": 0, "min": 0, "max": 0}
    lens = [len(c) for c in chunks]
    return {"count": len(chunks), "avg": sum(lens)/len(lens), "min": min(lens), "max": max(lens)}

@router.post("/chunk_preview")
async def chunk_preview_api(
    chunk_mode: str = Form("fixed_chars"),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
    text_column: Optional[str] = Form(None),

    # preview source
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),

    # sampling & pagination controls
    full_file: bool = Form(False),
    rows_to_sample: int = Form(25),
    page: int = Form(1),
    page_size: int = Form(100),
):
    """
    Preview chunking on pasted text or an uploaded file.
    - If full_file=True and file is provided, we concatenate the entire text column.
    - Otherwise we sample up to 'rows_to_sample' rows.
    - Results are paginated: 'page' (1-based) and 'page_size'.
    """
    if not text and not file:
        raise HTTPException(400, "Provide either 'text' or 'file'")

    source_text = None
    tmp_path: Optional[Path] = None

    try:
        if text:
            source_text = text[:HARD_LIMIT_CHARS]

        elif file:
            data = await file.read()

            # 1) Try extension from original filename
            import os
            ext = (os.path.splitext(file.filename or "")[1] or "").lower()

            # 2) If missing, try from content-type
            if not ext:
                ct = (getattr(file, "content_type", "") or "").lower()
                ct_map = {
                    "text/csv": ".csv",
                    "application/json": ".json",
                    "text/plain": ".txt",
                    "text/markdown": ".md",
                    "application/vnd.ms-excel": ".xls",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                }
                ext = ct_map.get(ct, "")

            # 3) If still missing, sniff magic bytes / structure
            if not ext:
                head = data[:8]
                if head.startswith(b"PK"):            # ZIP container → very likely XLSX
                    ext = ".xlsx"
                elif head.startswith(b"\xD0\xCF\x11\xE0"):  # OLE2 → old XLS
                    ext = ".xls"
                else:
                    # JSON?
                    s0 = data.lstrip()[:1]
                    if s0 in (b"{", b"["):
                        ext = ".json"
                    else:
                        # Heuristic CSV?
                        sample = data[:2048]
                        if b"," in sample and b"\n" in sample:
                            ext = ".csv"
                        else:
                            ext = ".txt"  # last resort: plain text

            # 4) Save with the inferred extension so read_any() recognizes it
            import tempfile
            from pathlib import Path
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)

            # 5) Read via your existing pipeline helpers
            df = normalize_df(read_any(tmp_path))
            if df.empty:
                raise HTTPException(400, f"Could not read any text from the file {file.filename or '(unnamed)'}")

            col = text_column or ("text" if "text" in df.columns else df.columns[0])

            # FULL FILE vs SAMPLE logic (keep whatever you implemented)
            series = df[col].astype(str)
            if full_file:
                joined = "\n\n".join(series.tolist())
                truncated = False
                if len(joined) > HARD_LIMIT_CHARS:
                    joined = joined[:HARD_LIMIT_CHARS]
                    truncated = True
                source_text = joined
            else:
                rows_to_sample = max(1, rows_to_sample)
                texts, total = [], 0
                truncated = False
                for s in series.head(rows_to_sample):
                    s = s.strip()
                    if not s:
                        continue
                    if total + len(s) + 2 > HARD_LIMIT_CHARS:
                        truncated = True
                        break
                    texts.append(s)
                    total += len(s) + 2
                source_text = "\n\n".join(texts)


        # Chunk the source text
        chunks_all = _chunk_text(source_text or "", chunk_mode, chunk_size, chunk_overlap)
        stats = _stats(chunks_all)

        # Pagination
        page = max(1, page)
        page_size = max(1, min(page_size, 500))
        start = (page - 1) * page_size
        end = start + page_size
        page_chunks = chunks_all[start:end]

        return {
            "params": {
                "mode": chunk_mode, "size": chunk_size, "overlap": chunk_overlap,
                "text_column": text_column, "full_file": bool(full_file),
                "rows_to_sample": rows_to_sample, "page": page, "page_size": page_size
            },
            "sample_chars": len(source_text or ""),
            "stats": stats,
            "total_chunks": stats["count"],
            "page_start": start,
            "page_end": min(end, stats["count"]),
            "chunks": page_chunks,
            "truncated_input": bool('truncated' in locals() and truncated),
        }

    finally:
        if tmp_path and tmp_path.exists():
            try: tmp_path.unlink()
            except Exception:
                pass
