import numpy as np
from typing import List, Dict


class InMemoryVectorStore:
    """In-memory vector store for documents and embeddings."""
    
    def __init__(self):
        self.docs = []
        self.embeddings = np.empty((0, 0))
    
    def add_documents(self, docs: List[Dict], embeddings: np.ndarray):
        """Add documents and their embeddings to the store."""
        self.docs.extend(docs)
        if self.embeddings.size == 0:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """Search for similar documents using cosine similarity."""
        if self.embeddings.size == 0:
            return []
        
        # Compute cosine similarity
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        
        query_embedding = query_embedding / query_norm
        similarities = np.dot(self.embeddings, query_embedding)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            if idx < len(self.docs):
                results.append({
                    "id": self.docs[idx]["id"],
                    "text": self.docs[idx]["text"],
                    "score": float(similarities[idx])
                })
        
        return results
