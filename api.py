
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging
import os
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions and queries",
    version="1.0.0"
)

# Add CORS middleware to allow requests from anywhere (since we're in Streamlit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
model_initializing = False

# Initialize model in background
def initialize_model():
    global model, model_initializing
    try:
        from src.utils.model_evaluation import ModelEvaluator
        logger.info("Starting model initialization")
        model_initializing = True
        
        # Create cache directory if it doesn't exist
        os.makedirs('cache', exist_ok=True)
        
        # Initialize model with cache directory
        model = ModelEvaluator('src/data/shl_full_catalog.csv', cache_dir='cache')
        logger.info("Model initialization completed successfully")
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}")
    finally:
        model_initializing = False

@app.get("/health")
async def health_check():
    """Health check endpoint that matches the specified format"""
    global model, model_initializing
    
    if model is not None:
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "model_status": "ready"
            }
        )
    elif model_initializing:
        return JSONResponse(
            status_code=200,
            content={
                "status": "initializing",
                "model_status": "loading"
            }
        )
    else:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_status": "failed"
            }
        )

@app.post("/recommend")
async def get_recommendations(
    query: str = Query(..., description="Job description or natural language query"),
    top_k: int = Query(10, description="Number of recommendations to return", ge=1, le=10)
):
    """Get assessment recommendations in the specified format"""
    global model
    
    if model is None and model_initializing:
        return JSONResponse(
            status_code=202,
            content={
                "status": "initializing",
                "message": "Model is still initializing, please try again in a few seconds"
            }
        )
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        start_time = time.time()
        result = model.evaluate_query(query, top_k=top_k, method='hybrid')
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Format recommendations according to specification
        recommendations = []
        for item in result['results']:
            test_types = [t.strip() for t in item['testTypes'].split(',') if t.strip()]
            # Extract numeric duration value
            duration_str = item['duration'].split()[0]  # Get first part (number)
            duration = int(duration_str) if duration_str.isdigit() else 0
            
            recommendations.append({
                "url": item['link'],
                "adaptive_support": "Yes" if item['adaptiveIRTSupport'].lower() == 'yes' else "No",
                "description": item['testName'],
                "duration": duration,
                "remote_support": "Yes" if item['remoteTestingSupport'].lower() == 'yes' else "No",
                "test_type": test_types
            })
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "recommendations": recommendations,
                "processing_time_ms": processing_time
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e),
                "recommendations": []
            }
        )

# Start model initialization when the app starts
@app.on_event("startup")
async def startup_event():
    import threading
    thread = threading.Thread(target=initialize_model)
    thread.daemon = True
    thread.start()
