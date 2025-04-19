
"""
API utility functions for the recommendation system
"""
import json
import requests
from typing import Dict, List, Any, Optional
import time

def format_api_response(recommendations: List[Dict[str, Any]], query: str, 
                        processing_time_ms: float) -> Dict[str, Any]:
    """
    Format API response according to the required specification
    
    Args:
        recommendations: List of recommendation dictionaries
        query: Original query string
        processing_time_ms: Processing time in milliseconds
        
    Returns:
        Formatted API response dictionary
    """
    return {
        "status": "success",
        "query": query,
        "processing_time_ms": processing_time_ms,
        "recommendations": [
            {
                "testName": rec["testName"],
                "link": rec["link"],
                "remoteTestingSupport": rec["remoteTestingSupport"],
                "adaptiveIRTSupport": rec["adaptiveIRTSupport"],
                "testTypes": rec["testTypes"],
                "duration": rec["duration"],
                "match_score": rec["score"]
            } for rec in recommendations
        ]
    }

def health_check() -> Dict[str, Any]:
    """
    Return API health status information
    
    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

def validate_query_params(query: Optional[str], top_k: Optional[int]) -> Dict[str, Any]:
    """
    Validate API query parameters
    
    Args:
        query: The search query string
        top_k: Number of results to return
        
    Returns:
        Validation result dictionary with success flag and error message if any
    """
    errors = []
    
    if not query or not query.strip():
        errors.append("Query parameter is required and cannot be empty")
    
    if top_k is not None:
        if not isinstance(top_k, int) or top_k < 1 or top_k > 10:
            errors.append("top_k parameter must be an integer between 1 and 10")
    
    return {
        "success": len(errors) == 0,
        "errors": errors
    }