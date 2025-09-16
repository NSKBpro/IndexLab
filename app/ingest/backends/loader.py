from pathlib import Path
from .faiss_flat import FaissFlat
from .faiss_ivf import FaissIVF

def create_backend(name: str, dim: int, build_params: dict):
    if name == "faiss_flat":
        return FaissFlat(dim)
    if name == "faiss_ivf":
        return FaissIVF(dim, nlist=int(build_params.get("nlist", 1024)), nprobe=int(build_params.get("nprobe", 10)))
    raise ValueError(f"Unknown backend: {name}")

def load_backend(name: str, path: Path, manifest: dict):
    if name == "faiss_flat":
        return FaissFlat.load(path)
    if name == "faiss_ivf":
        b = FaissIVF.load(path)
        b.nprobe = int(manifest.get("params", {}).get("nprobe", 10))
        return b
    if name == "hnswlib":
        return HnswLibBackend.load(path)
    raise ValueError(f"Unknown backend: {name}")
