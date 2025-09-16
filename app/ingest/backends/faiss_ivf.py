from pathlib import Path
import faiss
import numpy as np

class FaissIVF:
    def __init__(self, dim: int, nlist: int = 1024, nprobe: int = 10):
        quant = faiss.IndexFlatIP(dim)
        self.idx = faiss.IndexIVFFlat(quant, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        self.nprobe = nprobe
        self._is_trained = False

    def add(self, embeddings: np.ndarray) -> None:
        if not self._is_trained:
            self.idx.train(embeddings)
            self._is_trained = True
        self.idx.add(embeddings)

    def search(self, query: np.ndarray, k: int):
        self.idx.nprobe = self.nprobe
        return self.idx.search(query, k)

    def save(self, path: Path) -> None:
        faiss.write_index(self.idx, str(path))

    @classmethod
    def load(cls, path: Path):
        idx = faiss.read_index(str(path))
        obj = cls(idx.d)
        obj.idx = idx
        obj._is_trained = True
        return obj
