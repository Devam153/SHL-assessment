from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import time
import uvicorn
import pandas as pd

from src.utils.model_evaluation import ModelEvaluator
from src.utils.api_utils import format_api_response, health_check, validate_query_params

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions and queries",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and data
model = None

@app.on_event("startup")
async def startup_event():
    """Initialize model and load data on startup"""
    global model
    try:
        model = ModelEvaluator('src/data/shl_full_catalog.csv')
    except Exception as e:
        print(f"Error initializing model: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "SHL Assessment Recommender API",
        "version": "1.0.0",
        "endpoints": [
            "/api/health",
            "/api/recommendations"
        ]
    }

@app.get("/api/health")
async def api_health():
    """Health check endpoint"""
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": "Model not initialized"}
        )
    return health_check()

@app.get("/api/recommendations")
async def get_recommendations(
    query: str = Query(..., description="Job description or natural language query"),
    top_k: int = Query(5, description="Number of recommendations to return (1-10)", ge=1, le=10),
    method: str = Query("hybrid", description="Search method to use (semantic, tfidf, or hybrid)")
):
    """
    Get assessment recommendations based on a query
    
    - **query**: Job description or natural language query
    - **top_k**: Number of recommendations to return (1-10)
    - **method**: Search method (semantic, tfidf, or hybrid)
    
    Returns a list of recommended assessments with scores and metadata
    """
    global model
    
    # Validate query parameters
    validation = validate_query_params(query, top_k)
    if not validation["success"]:
        raise HTTPException(status_code=400, detail=validation["errors"])
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Get recommendations
        result = model.evaluate_query(query, top_k=top_k, method=method)
        
        # Format API response
        response = format_api_response(
            recommendations=result["results"],
            query=query,
            processing_time_ms=result["processing_time_ms"]
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.get("/api/methods")
async def get_available_methods():
    """Get information about available search methods"""
    return {
        "methods": [
            {
                "id": "semantic",
                "name": "Semantic Search",
                "description": "Uses sentence transformers to find semantically similar assessments"
            },
            {
                "id": "tfidf",
                "name": "TF-IDF Search",
                "description": "Uses TF-IDF vectorization to find keyword matches"
            },
            {
                "id": "hybrid",
                "name": "Hybrid Search",
                "description": "Combines semantic and TF-IDF approaches for better results"
            }
        ]
    }

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
