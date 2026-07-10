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
