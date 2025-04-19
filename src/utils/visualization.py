
"""
Visualization utilities for displaying model evaluation results and recommendations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional

def plot_benchmark_comparison(benchmark_df: pd.DataFrame, metric: str = 'mean_recall_at_k') -> None:
    """
    Plot performance comparison of different methods
    
    Args:
        benchmark_df: DataFrame with benchmark results
        metric: Metric to plot ('mean_recall_at_k', 'map_at_k', or 'avg_processing_time_ms')
    """
    if benchmark_df.empty:
        st.warning("No benchmark data available to plot.")
        return
        
    # Check for NaN values and replace with 0
    benchmark_df = benchmark_df.fillna(0)
    
    # Special handling for processing time
    is_time_metric = metric == 'avg_processing_time_ms'
    
    # Check if all metric values are effectively zero (less than a small threshold)
    if not is_time_metric and (benchmark_df[metric] < 0.00001).all():
        st.warning(f"No meaningful data to display for {metric}. All values are effectively zero.")
        return

    # Create column with formatted values for hover
    if is_time_metric:
        benchmark_df['hover_value'] = benchmark_df[metric].apply(lambda x: f"{x:.2f} ms")
        title_text = "Processing Time Comparison by Method"
        y_axis_title = "Time (milliseconds)"
    else:
        benchmark_df['hover_value'] = benchmark_df[metric].apply(lambda x: f"{x:.4f}")
        title_text = f"Performance Comparison by Method ({metric.replace('_', ' ').title()})"
        y_axis_title = metric.replace('_', ' ').title()

    # Color mapping for consistent colors across charts
    color_map = {'semantic': '#1F77B4', 'tfidf': '#36A2EB', 'hybrid': '#FF6384'}
    
    # Create the bar chart
    fig = px.bar(
        benchmark_df, 
        x='method', 
        y=metric, 
        title=title_text,
        color='method',
        color_discrete_map=color_map,
        hover_data={
            'method': True,
            'hover_value': True,
            'queries_evaluated': True,
            metric: False
        },
        labels={
            'method': 'Search Method',
            'hover_value': 'Value'
        }
    )
    
    # Add labels for values on top of each bar
    if is_time_metric:
        fig.update_traces(
            texttemplate='%{y:.1f} ms', 
            textposition='outside'
        )
    else:
        fig.update_traces(
            texttemplate='%{y:.3f}', 
            textposition='outside'
        )
    
    fig.update_layout(
        xaxis_title='Method',
        yaxis_title=y_axis_title,
        legend_title='Search Method',
        height=500
    )
    
    # For time metrics, we might want to add a y-axis grid
    if is_time_metric:
        fig.update_layout(
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add explanation text below the chart based on metric type
    if metric == 'mean_recall_at_k':
        st.info("📊 **Mean Recall@K** measures the average proportion of relevant items that are successfully retrieved in the top K results.")
    elif metric == 'map_at_k':
        st.info("📊 **Mean Average Precision@K** measures both precision and ranking quality of the search results.")
    elif metric == 'avg_processing_time_ms':
        st.info("⏱️ **Average Processing Time** shows how long each method takes to process a query in milliseconds.")

def plot_test_type_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of test types in the catalog
    
    Args:
        df: DataFrame with test type information
    """
    # Determine which column name to use for test types
    test_types_col = None
    if 'Test Types' in df.columns:
        test_types_col = 'Test Types'
    elif 'testTypes' in df.columns:
        test_types_col = 'testTypes'
    
    if test_types_col is None:
        st.warning("Test Types column not found in dataset. Cannot display test type distribution.")
        return
        
    # Extract test types
    all_types = []
    
    for types_str in df[test_types_col]:
        if isinstance(types_str, str):
            types = [t.strip() for t in types_str.split(',')]
            all_types.extend(types)
    
    # Count occurrences
    if not all_types:
        st.warning("No test type data found in dataset.")
        return
        
    type_counts = pd.Series(all_types).value_counts().reset_index()
    type_counts.columns = ['Test Type', 'Count']
    
    # Create plot
    fig = px.bar(
        type_counts, 
        x='Test Type', 
        y='Count',
        title='Distribution of Test Types in Assessment Catalog',
        color='Test Type'
    )
    
    fig.update_layout(
        xaxis_title='Test Type',
        yaxis_title='Number of Assessments',
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_duration_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of test durations
    
    Args:
        df: DataFrame with duration information
    """
    # Determine which column name to use for duration
    duration_col = None
    if 'Duration' in df.columns:
        duration_col = 'Duration'
    elif 'duration' in df.columns:
        duration_col = 'duration'
    
    if duration_col is None:
        st.info("Duration column not found in dataset. Skipping duration distribution analysis.")
        return
        
    # Extract duration values
    durations = []
    
    for duration in df[duration_col]:
        if isinstance(duration, str):
            # Handle ranges like "35-40 min"
            parts = duration.replace('min', '').strip().split('-')
            try:
                # Take the first number if range, or the only number
                duration_value = int(parts[0])
                durations.append(duration_value)
            except ValueError:
                pass
    
    # Create histogram
    if not durations:
        st.warning("No valid duration data found in dataset.")
        return
        
    fig = px.histogram(
        x=durations,
        nbins=10,
        title='Distribution of Assessment Durations',
        labels={'x': 'Duration (minutes)', 'y': 'Count'},
        color_discrete_sequence=['#36A2EB']
    )
    
    fig.update_layout(
        xaxis_title='Duration (minutes)',
        yaxis_title='Number of Assessments',
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_remote_adaptive_support(df: pd.DataFrame) -> None:
    """
    Visualize remote testing and adaptive support distribution
    
    Args:
        df: DataFrame with support information
    """
    # Determine which column names to use
    remote_col = None
    adaptive_col = None
    
    if 'Remote Testing' in df.columns:
        remote_col = 'Remote Testing'
    elif 'remoteTestingSupport' in df.columns:
        remote_col = 'remoteTestingSupport'
        
    if 'Adaptive/IRT' in df.columns:
        adaptive_col = 'Adaptive/IRT'
    elif 'adaptiveIRTSupport' in df.columns:
        adaptive_col = 'adaptiveIRTSupport'
    
    if remote_col is None or adaptive_col is None:
        st.warning("Required columns for testing support analysis not found in dataset.")
        return
    
    try:
        # Remote Testing Support
        remote_count = df[remote_col].str.lower().eq('yes').sum()
        remote_data = {'Category': ['Supports Remote', 'No Remote Support'], 
                      'Count': [remote_count, len(df) - remote_count]}
        
        # Adaptive/IRT Support
        adaptive_count = df[adaptive_col].str.lower().eq('yes').sum()
        adaptive_data = {'Category': ['Supports Adaptive', 'No Adaptive Support'], 
                        'Count': [adaptive_count, len(df) - adaptive_count]}
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.pie(
                remote_data,
                values='Count',
                names='Category',
                title='Remote Testing Support',
                color_discrete_sequence=['#36A2EB', '#FFCE56']
            )
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.pie(
                adaptive_data,
                values='Count',
                names='Category',
                title='Adaptive/IRT Support',
                color_discrete_sequence=['#FF6384', '#4BC0C0']
            )
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating support visualizations: {str(e)}")

def display_recommendation_details(recommendation: Dict[str, Any]) -> None:
    """
    Display detailed information for a single recommendation
    
    Args:
        recommendation: Dictionary with recommendation details
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

def add_search_method_explanation() -> None:
    """
    Add explanation about the different search methods used in the application
    """
    with st.expander("📚 Understanding Search Methods"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Semantic Search")
            st.markdown("""
            - Uses **neural embeddings** to capture meaning
            - Better at understanding context and synonyms
            - Can find assessments that are semantically related even when words don't match exactly
            - Powered by the Sentence Transformer model
            """)
            
        with col2:
            st.markdown("### TF-IDF Search")
            st.markdown("""
            - Uses **term frequency-inverse document frequency**
            - Good at matching specific keywords and technical terms
            - Works well when exact terminology is important
            - Based on statistical word occurrence patterns
            """)
            
        with col3:
            st.markdown("### Hybrid Search")
            st.markdown("""
            - Combines both **semantic and TF-IDF** approaches
            - Balances meaning-based and keyword-based matching
            - Often provides the best overall performance
            - Default recommendation method in the system
            """)
            
        st.info("The benchmark results show how each method performs on sample queries. Higher recall and precision values indicate better performance, while lower processing time indicates faster results.")
