from fastapi import FastAPI
from pydantic import BaseModel

class Query(BaseModel):
    q: str

app = FastAPI(title="Entangled Retrievals - Demo")

@app.get("/")
async def root():
    return {"message": "Entangled Retrievals API"}

@app.post("/retrieve")
async def retrieve(query: Query):
    # Placeholder retrieval: replace with real vector search later
    return {
        "query": query.q,
        "results": [
            {"id": "doc1", "score": 0.92, "text": f"Example result for '{query.q}'"},
            {"id": "doc2", "score": 0.78, "text": "Related example document"}
        ]
    }
