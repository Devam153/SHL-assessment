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
    match_percentage = int(recommendation.get('score', 0) * 100)
    
    # Color coding based on match score
    match_color = (
        "green" if match_percentage >= 80 
        else "orange" if match_percentage >= 60 
        else "gray"
    )
    
    # Format duration consistently if available
    duration = recommendation.get('duration', 'Not specified')
    if duration and isinstance(duration, str):
        # Handle special cases like "Variable" or "Untimed"
        special_durations = ["variable", "untimed", "varies"]
        duration_lower = duration.lower()
        
        if any(sd in duration_lower for sd in special_durations):
            # Keep special values as is, but with consistent formatting
            if "variable" in duration_lower:
                duration = "Variable"
            elif "untimed" in duration_lower:
                duration = "Untimed"
            elif "varies" in duration_lower:
                duration = "Varies"
        # Handle "max" durations
        elif "max" in duration_lower:
            max_match = re.search(r'max\s*(\d+)', duration_lower)
            if max_match:
                duration = f"Max {max_match.group(1)} min"
        else:
            # Make sure numeric durations end with "min" for consistency
            if not duration_lower.endswith('min'):
                duration = f"{duration} min"
    
    # Get test name from the appropriate field - handle both "testName" and "Test Name"
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

# Function to deduplicate recommendations based on test name
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