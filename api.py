from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
from search_engine import RecipeSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Recipe Search API with SBERT and Autoencoder")

class SearchResponse(BaseModel):
    results: List[Dict]
    message: str

# Initialize search engine
search_engine = RecipeSearchEngine()

@app.on_event("startup")
async def startup_event():
    search_engine.prepare_data('recipes.csv')
    search_engine.visualize_embeddings('embeddings_viz')

@app.get("/search", response_model=SearchResponse)
async def search(query: str = Query(..., description="Search query text"), 
                limit: Optional[int] = Query(10, description="Number of results to return")):
    try:
        results = search_engine.search(query, top_k=limit)
        return SearchResponse(results=results, message="OK")
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=6352)