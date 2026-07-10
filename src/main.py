from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from .embedder import Embedder
from .store import InMemoryVectorStore
from .persistent_store import save_index, load_index


class Query(BaseModel):
    q: str
    k: Optional[int] = 5


class Document(BaseModel):
    id: str
    text: str


app = FastAPI(title="Entangled Retrievals - Demo")

embedder = Embedder()
store = InMemoryVectorStore()


@app.get("/")
async def root():
    return {"message": "Entangled Retrievals API"}


@app.post("/index")
async def index(docs: List[Document]):
    texts = [d.text for d in docs]
    embedder.fit(texts)
    embs = embedder.encode(texts)
    docs_dicts = [{"id": d.id, "text": d.text} for d in docs]
    store.add_documents(docs_dicts, embs)
    return {"indexed": len(docs)}


@app.post("/retrieve")
async def retrieve(query: Query):
    if store.embeddings.size == 0:
        raise HTTPException(status_code=404, detail="No documents indexed")
    q_emb = embedder.encode([query.q])[0]
    results = store.search(q_emb, k=query.k)
    return {"query": query.q, "results": results}


@app.get("/docs")
async def list_docs():
    return {"count": len(store.docs), "docs": store.docs}


@app.post("/save")
async def save(path: str = "data/index.joblib"):
    if store.embeddings.size == 0:
        raise HTTPException(status_code=400, detail="No index to save")
    save_index(path, store.docs, store.embeddings)
    return {"saved": path}


@app.post("/load")
async def load(path: str = "data/index.joblib"):
    try:
        docs, embs = load_index(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index file not found")
    if embs is None or len(embs) == 0:
        raise HTTPException(status_code=400, detail="Loaded index contains no embeddings")
    global store
    store = InMemoryVectorStore()
    store.add_documents(docs, embs)
    return {"loaded": len(docs)}
