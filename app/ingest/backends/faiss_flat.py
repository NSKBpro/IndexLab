from pathlib import Path
import faiss
import numpy as np

class FaissFlat:
    def __init__(self, dim: int):
        self.idx = faiss.IndexFlatIP(dim)

    def add(self, embeddings: np.ndarray) -> None:
        self.idx.add(embeddings)

    def search(self, query: np.ndarray, k: int):
        return self.idx.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self.idx, str(path))

    @classmethod
    def load(cls, path: Path):
        idx = faiss.read_index(str(path))
        obj = cls(idx.d)
        obj.idx = idx
        return obj
