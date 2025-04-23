from fastapi import FastAPI, Query, HTTPException, Request, Response
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
import os
import json
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SHL Assessment Recommender API",
    description="API for recommending SHL assessments based on job descriptions and queries",
    version="1.0.0"
)

def load_metadata():
    return pd.read_csv("src/data/shl_full_catalog_with_duration_desc.csv")

df_meta = load_metadata()

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

# Test type mapping
TEST_TYPE_MAP = {
    "K": "Knowledge & Skills",
    "B": "Behavioral",
    "P": "Personality", 
    "C": "Cognitive",
    "A": "Aptitude",
    "S": "Situational",
    "T": "Technical",
    "N": "Numerical",
    "L": "Leadership",
    "D": "Decision Making",
    "E": "Emotional Intelligence"
}

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

# Modified to accept both GET and POST requests
@app.get("/recommend")
@app.post("/recommend")
async def get_recommendations(
    query: str = Query(..., description="Job description or natural language query"),
    top_k: int = Query(10, description="Number of recommendations to return", ge=1, le=10),
    format: str = Query("json", description="Response format: json or html")
):
    """Get assessment recommendations in the specified format"""
    global model

    if model is None and model_initializing:
        response_obj = {
            "status": "initializing",
            "message": "Model is still initializing, please try again in a few seconds"
        }
        pretty_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        return Response(content=pretty_json, media_type="application/json", status_code=202)
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        start_time = time.time()
        result = model.evaluate_query(query, top_k=top_k, method='hybrid')
        processing_time_ms = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        recommendations = []
        for item in result["results"][:top_k]:

            link = item.get("link", "")
                        
            meta = df_meta[df_meta["Link"] == link]
            if meta.empty:
                continue
            row = meta.iloc[0]

            description = row.get("Description", "")
            
            # Fix for duration handling - safely extract duration number
            duration = 0
            duration_str = row.get("Duration", "")
            if isinstance(duration_str, str) and duration_str:
                import re
                duration_match = re.search(r'\d+', duration_str)
                if duration_match:
                    duration = int(duration_match.group())

            # Handle test types
            raw_types = row.get("Test Types", "")
            test_types = []
            if isinstance(raw_types, str):
                test_types = [
                    TEST_TYPE_MAP.get(t.strip(), t.strip()) for t in raw_types.split(",") if t.strip()
                ]

            recommendations.append({
                "url": link,
                "adaptive_support": "Yes" if item.get("adaptiveIRTSupport", "").lower() == "yes" else "No",
                "description": description,
                "duration": duration,
                "remote_support": "Yes" if item.get("remoteTestingSupport", "").lower() == "yes" else "No",
                "test_type": test_types 
            })
        
        response_obj = {
            "status": "success",
            "recommended_assessments": recommendations,
            "processing_time_ms": processing_time_ms
        }

        pretty_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        return Response(content=pretty_json, media_type="application/json", status_code=200)
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        response_obj = {
            "status": "error",
            "message": str(e),
            "recommended_assessments": []
        }
        pretty_json = json.dumps(response_obj, indent=2, ensure_ascii=False)
        return Response(content=pretty_json, media_type="application/json", status_code=500)

@app.on_event("startup")
async def startup_event():
    import threading
    thread = threading.Thread(target=initialize_model)
    thread.daemon = True
    thread.start()