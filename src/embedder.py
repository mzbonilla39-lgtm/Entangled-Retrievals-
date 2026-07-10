import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder:
    """Simple TF-IDF based embedder for text documents."""
    
    def __init__(self):
        self.vectorizer = None
    
    def fit(self, texts):
        """Fit the embedder on a collection of texts."""
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.vectorizer.fit(texts)
    
    def encode(self, texts):
        """Encode texts into embedding vectors."""
        if self.vectorizer is None:
            raise ValueError("Embedder not fitted. Call fit() first.")
        return self.vectorizer.transform(texts).toarray()
