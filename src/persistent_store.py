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
import joblib
import numpy as np
from typing import List, Dict, Tuple, Optional


def save_index(path: str, docs: List[Dict], embeddings: np.ndarray) -> None:
    """Save documents and embeddings to disk using joblib."""
    data = {
        "docs": docs,
        "embeddings": embeddings
    }
    joblib.dump(data, path)


def load_index(path: str) -> Tuple[List[Dict], Optional[np.ndarray]]:
    """Load documents and embeddings from disk using joblib."""
    data = joblib.load(path)
    return data.get("docs", []), data.get("embeddings", None)
