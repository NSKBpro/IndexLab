from rank_bm25 import BM25Okapi
from typing import List, Tuple

def bm25_build(corpus_texts: list[str]) -> BM25Okapi:
    tokenized = [t.lower().split() for t in corpus_texts]
    return BM25Okapi(tokenized)

def bm25_search(bm25: BM25Okapi, query: str, k: int) -> List[Tuple[int, float]]:
    tokenized = query.lower().split()
    scores = bm25.get_scores(tokenized)
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]
    return ranked

def rrf_fuse(vec_hits: list[Tuple[int,float]], bm_hits: list[Tuple[int,float]], k: int, k_rrf: int = 60) -> list[Tuple[int, float]]:
    rank_map = {}
    for r, (i, _) in enumerate(vec_hits):
        rank_map[i] = rank_map.get(i, 0.0) + 1.0 / (k_rrf + r + 1)
    for r, (i, _) in enumerate(bm_hits):
        rank_map[i] = rank_map.get(i, 0.0) + 1.0 / (k_rrf + r + 1)
    fused = sorted(rank_map.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(i, s) for i, s in fused]
