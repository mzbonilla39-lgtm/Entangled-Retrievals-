"""Entangled Retrievals - Main API Module."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import logging
import sys
import os

from .embedder import Embedder
from .store import InMemoryVectorStore
from .persistent_store import save_index, load_index

# Add parent directory to path for config imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config.settings import settings
except ImportError:
    # Fallback settings if config module not available
    class FallbackSettings:
        app_name = "Entangled Retrievals"
        app_version = "1.0.0"
        env = "development"
        host = "0.0.0.0"
        port = 8000
        reload = True
        log_level = "INFO"
        cors_origins = ["http://localhost", "http://localhost:3000", "http://localhost:8000"]
    
    settings = FallbackSettings()

# Configure logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Safe, structured third-party personal property retrieval service API"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class Query(BaseModel):
    """Query model for document retrieval."""
    q: str
    k: Optional[int] = 5


class Document(BaseModel):
    """Document model for indexing."""
    id: str
    text: str


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    env: str


# Initialize embedder and vector store
embedder = Embedder()
store = InMemoryVectorStore()


# Health & Status Endpoints
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for monitoring."""
    try:
        return {
            "status": "healthy",
            "version": settings.app_version,
            "env": settings.env
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.get("/", tags=["Status"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Entangled Retrievals API",
        "version": settings.app_version,
        "status": "operational"
    }


# Indexing Endpoints
@app.post("/index", tags=["Indexing"])
async def index(docs: List[Document]):
    """Index documents with embeddings for retrieval."""
    try:
        if not docs:
            logger.warning("Empty document list provided for indexing")
            return {"indexed": 0, "warning": "No documents provided"}
        
        texts = [d.text for d in docs]
        logger.info(f"Indexing {len(docs)} documents")
        
        embedder.fit(texts)
        embs = embedder.encode(texts)
        docs_dicts = [{"id": d.id, "text": d.text} for d in docs]
        store.add_documents(docs_dicts, embs)
        
        logger.info(f"Successfully indexed {len(docs)} documents")
        return {
            "indexed": len(docs),
            "total_documents": len(store.docs)
        }
    except Exception as e:
        logger.error(f"Indexing error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Indexing failed: {str(e)}")


# Retrieval Endpoints
@app.post("/retrieve", tags=["Retrieval"])
async def retrieve(query: Query):
    """Retrieve similar documents based on query."""
    try:
        if store.embeddings.size == 0:
            logger.warning("Retrieval attempted on empty index")
            raise HTTPException(
                status_code=404,
                detail="No documents indexed. Please index documents first."
            )
        
        # Validate query is not empty
        if not query.q or len(query.q.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Validate query length to prevent resource exhaustion
        max_query_length = 5000
        if len(query.q) > max_query_length:
            raise HTTPException(
                status_code=400,
                detail=f"Query exceeds maximum length of {max_query_length} characters"
            )
        
        # Validate k parameter
        if query.k is not None and query.k <= 0:
            raise HTTPException(status_code=400, detail="k must be greater than 0")
        
        logger.info(f"Processing query: {query.q[:50]}...")
        q_emb = embedder.encode([query.q])[0]
        results = store.search(q_emb, k=query.k)
        
        logger.info(f"Query returned {len(results)} results")
        return {
            "query": query.q,
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Retrieval error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Retrieval failed: {str(e)}")


# Document Management Endpoints
@app.get("/docs", tags=["Management"])
async def list_docs():
    """List all indexed documents."""
    try:
        return {
            "count": len(store.docs),
            "docs": store.docs
        }
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=400, detail="Failed to list documents")


@app.post("/save", tags=["Management"])
async def save(path: str = "data/index.joblib"):
    """Save current index to disk."""
    try:
        if store.embeddings.size == 0:
            raise HTTPException(status_code=400, detail="No index to save")
        
        logger.info(f"Saving index to {path}")
        save_index(path, store.docs, store.embeddings)
        
        logger.info(f"Index saved successfully to {path}")
        return {
            "saved": path,
            "document_count": len(store.docs)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Save error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save index: {str(e)}")


@app.post("/load", tags=["Management"])
async def load(path: str = "data/index.joblib"):
    """Load index from disk."""
    try:
        logger.info(f"Loading index from {path}")
        docs, embs = load_index(path)
        
        if embs is None or len(embs) == 0:
            raise HTTPException(status_code=400, detail="Loaded index contains no embeddings")
        
        global store
        store = InMemoryVectorStore()
        store.add_documents(docs, embs)
        
        logger.info(f"Index loaded successfully with {len(docs)} documents")
        return {
            "loaded": len(docs),
            "path": path
        }
    except FileNotFoundError:
        logger.error(f"Index file not found: {path}")
        raise HTTPException(status_code=404, detail=f"Index file not found: {path}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Load error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load index: {str(e)}")


# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return {
        "error": "Internal server error",
        "status_code": 500
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_level=settings.log_level.lower()
    )
