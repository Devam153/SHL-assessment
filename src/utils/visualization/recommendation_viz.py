
"""
Visualization utilities for displaying recommendations
"""
import streamlit as st
from typing import Dict, Any

def display_recommendation_details(recommendation: Dict[str, Any]) -> None:
    """
    Display detailed information for a single recommendation
    """
    st.markdown(f"### {recommendation['testName']}")
    
    # Create two columns for the details
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Match Score:** {recommendation['score']*100:.1f}%")
        st.markdown(f"**Test Types:** {recommendation.get('testTypes', 'Not specified')}")
        st.markdown(f"**Duration:** {recommendation.get('duration', 'Not specified')}")
        
    with col2:
        st.markdown(f"**Remote Testing:** {recommendation.get('remoteTestingSupport', 'Not specified')}")
        st.markdown(f"**Adaptive/IRT:** {recommendation.get('adaptiveIRTSupport', 'Not specified')}")
        st.markdown(f"[View Assessment Details]({recommendation['link']})")
