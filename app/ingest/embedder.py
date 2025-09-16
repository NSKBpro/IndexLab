from sentence_transformers import SentenceTransformer
import numpy as np
from typing import Dict

_model_cache: Dict[str, SentenceTransformer] = {}

def get_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]

def embed_texts(texts: list[str], model_name: str, normalize: bool) -> np.ndarray:
    print(texts)
    model = get_model(model_name)
    print(model)
    embs = model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
    print(embs)
    return embs.astype("float32")
