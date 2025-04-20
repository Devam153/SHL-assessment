
"""
Visualization utilities for displaying recommendations
"""
import streamlit as st
import re
from typing import Dict, Any

def display_recommendation_details(recommendation: Dict[str, Any]) -> None:
    """
    Display detailed information for a single recommendation
    """
    raw_score = recommendation.get('score', 0)

    boosted_score = min(1.0, raw_score * 1.8 if raw_score < 0.5 else 0.9 + (raw_score - 0.5) * 0.2)
    match_percentage = int(boosted_score * 100)
    
    match_color = (
        "green" if match_percentage >= 75  
        else "orange" if match_percentage >= 55  
        else "gray"
    )
    
    duration = recommendation.get('duration', 'Not specified')
    if duration and isinstance(duration, str):
        special_durations = ["variable", "untimed", "varies"]
        duration_lower = duration.lower()
        
        if any(sd in duration_lower for sd in special_durations):
            if "variable" in duration_lower:
                duration = "Variable"
            elif "untimed" in duration_lower:
                duration = "Untimed"
            elif "varies" in duration_lower:
                duration = "Varies"
        elif "max" in duration_lower:
            max_match = re.search(r'max\s*(\d+)', duration_lower)
            if max_match:
                duration = f"Max {max_match.group(1)} min"
        else:
            if not duration_lower.endswith('min'):
                duration = f"{duration} min"
    
    test_name = recommendation.get('testName', recommendation.get('Test Name', 'Unknown Test'))
    test_types = recommendation.get('testTypes', recommendation.get('Test Types', 'Not specified'))
    remote_testing = recommendation.get('remoteTestingSupport', recommendation.get('Remote Testing', 'Not specified'))
    adaptive_testing = recommendation.get('adaptiveIRTSupport', recommendation.get('Adaptive/IRT', 'Not specified'))
    link = recommendation.get('link', recommendation.get('Link', '#'))
    
    st.markdown(f"""
    <div class="card">
        <h3>{test_name} 
            <span style="float:right; color:{match_color}; font-weight:bold;">
                {match_percentage}% Match
            </span>
        </h3>
        <p><strong>Test Types:</strong> {test_types}</p>
        <p><strong>Duration:</strong> {duration}</p>
        <p><strong>Remote Testing:</strong> {remote_testing}</p>
        <p><strong>Adaptive Testing:</strong> {adaptive_testing}</p>
        <p><a href="{link}" target="_blank">View Assessment Details</a></p>
    </div>
    """, unsafe_allow_html=True)

def deduplicate_recommendations(recommendations):
    """
    Remove duplicate recommendations based on test name
    """
    unique_recommendations = []
    seen_test_names = set()
    
    for rec in recommendations:
        test_name = rec.get('testName', rec.get('Test Name', ''))
        if test_name and test_name not in seen_test_names:
            seen_test_names.add(test_name)
            unique_recommendations.append(rec)
    
    return unique_recommendations