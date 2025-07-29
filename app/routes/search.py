"""
Semantic search API routes
"""
from typing import List
from fastapi import APIRouter, HTTPException
from app.models.schemas import SemanticSearchQuery, SemanticSearchResult
from app.services.redis_service import redis_service
from app.services.ai_service import ai_service

router = APIRouter()

@router.post("/", response_model=List[SemanticSearchResult])
async def semantic_search(query: SemanticSearchQuery):
    """Perform semantic search across stored content"""
    try:
        # Generate embedding for the query
        query_embedding = ai_service.generate_embedding(query.query)

        # Perform search
        results = redis_service.semantic_search(query_embedding, limit=query.limit)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

@router.post("/index")
async def index_content(content: str, device_id: str, metadata: dict = None):
    """Index content for semantic search"""
    try:
        # Generate embedding
        embedding = ai_service.generate_embedding(content)

        # Prepare metadata
        search_metadata = {
            "device_id": device_id,
            "content_type": "device_info",
            **(metadata or {})
        }

        # Store in Redis
        success = redis_service.store_embedding(content, embedding, search_metadata)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to index content")

        return {"message": "Content indexed successfully", "content_length": len(content)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

@router.get("/stats")
async def get_search_stats():
    """Get search index statistics"""
    try:
        stats = redis_service.get_stats()
        return {
            "indexed_documents": stats.get("embedding_count", 0),
            "redis_memory": stats.get("used_memory_human", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")
