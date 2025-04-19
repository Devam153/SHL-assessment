
"""
Visualization utilities for benchmark results
"""
import pandas as pd
import streamlit as st
import plotly.express as px
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
        
    benchmark_df = benchmark_df.fillna(0)
    
    is_time_metric = metric == 'avg_processing_time_ms'
    
    if not is_time_metric and (benchmark_df[metric] < 0.00001).all():
        st.warning(f"No meaningful data to display for {metric}. All values are effectively zero.")
        return

    if is_time_metric:
        benchmark_df['hover_value'] = benchmark_df[metric].apply(lambda x: f"{x:.2f} ms")
        title_text = "Processing Time Comparison by Method"
        y_axis_title = "Time (milliseconds)"
    else:
        benchmark_df['hover_value'] = benchmark_df[metric].apply(lambda x: f"{x:.4f}")
        title_text = f"Performance Comparison by Method ({metric.replace('_', ' ').title()})"
        y_axis_title = metric.replace('_', ' ').title()

    color_map = {'semantic': '#1F77B4', 'tfidf': '#36A2EB', 'hybrid': '#FF6384'}
    
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
    
    if is_time_metric:
        fig.update_layout(
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(0,0,0,0.1)'
            )
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if metric == 'mean_recall_at_k':
        st.info("📊 **Mean Recall@K** measures the average proportion of relevant items that are successfully retrieved in the top K results.")
    elif metric == 'map_at_k':
        st.info("📊 **Mean Average Precision@K** measures both precision and ranking quality of the search results.")
    elif metric == 'avg_processing_time_ms':
        st.info("⏱️ **Average Processing Time** shows how long each method takes to process a query in milliseconds.")
