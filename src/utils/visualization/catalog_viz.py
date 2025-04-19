
"""
Visualization utilities for catalog analysis
"""
import pandas as pd
import streamlit as st
import plotly.express as px

def plot_test_type_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of test types in the catalog
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

def plot_remote_adaptive_support(df: pd.DataFrame) -> None:
    """
    Visualize remote testing and adaptive support distribution
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
