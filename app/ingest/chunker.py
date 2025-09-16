import re
from typing import Iterable
import pandas as pd

def concat_row(row) -> str:
    return " | ".join(str(v) for v in row.values if isinstance(v, (str, int, float)) and str(v).strip())

def iter_rows(df: pd.DataFrame, text_column: str | None) -> Iterable[tuple[str, str]]:
    for i, row in df.iterrows():
        text = str(row[text_column]) if text_column and text_column in df.columns else concat_row(row)
        if text.strip():
            yield (str(i), text)

def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    out, i, n = [], 0, len(text)
    while i < n:
        j = min(i + size, n)
        out.append(text[i:j])
        if j == n: break
        i = max(j - overlap, i + 1)
    return out

_sentence_re = re.compile(r'(?<=[\.!?])\s+')

def chunk_sentences(text: str, size: int, overlap: int) -> list[str]:
    sents = _sentence_re.split(text)
    out, cur = [], ""
    for s in sents:
        if len(cur) + len(s) + 1 <= size:
            cur = (cur + " " + s).strip()
        else:
            if cur: out.append(cur)
            cur = s
    if cur: out.append(cur)
    if overlap > 0 and len(out) > 1:
        out = [out[0]] + [out[i-1][-overlap:] + out[i] for i in range(1, len(out))]
    return out

def chunk_by_headings(text: str, size: int, overlap: int) -> list[str]:
    parts = re.split(r'\n\s*(#+|\<h[1-3]\>|\</h[1-3]\>)', text)
    joined, buf = [], ""
    for p in parts:
        if p and p.strip().startswith(("#","<h","</h")):
            if buf: joined.append(buf.strip()); buf=""
        else:
            buf += ("\n" + p)
    if buf: joined.append(buf.strip())
    out = []
    for sect in joined:
        out.extend(chunk_fixed(sect, size, overlap))
    return out
