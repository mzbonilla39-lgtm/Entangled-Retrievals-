from typing import List, Dict, Any
import joblib
import numpy as np
from pathlib import Path


def save_index(path: str, docs: List[Dict[str, Any]], embeddings: np.ndarray):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"docs": docs, "embeddings": embeddings}
    joblib.dump(payload, str(p))


def load_index(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    payload = joblib.load(str(p))
    docs = payload.get("docs", [])
    embeddings = payload.get("embeddings", None)
    if embeddings is not None:
        embeddings = np.asarray(embeddings, dtype=float)
    return docs, embeddings
