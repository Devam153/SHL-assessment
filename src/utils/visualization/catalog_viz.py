"""
Visualization utilities for catalog analysis
"""
import pandas as pd 
import streamlit as st
import plotly.express as px
import re
from collections import Counter

def clean_duration(duration_str: str) -> int | None:
    """
    Clean duration string and convert to integer minutes.
    Returns None if duration is not a valid number.
    """
    if not isinstance(duration_str, str):
        return None
        
    # Normalize the string
    duration = duration_str.lower().strip()
    
    # Handle special cases first
    if duration in ['untimed', 'variable', '']:
        return None
    
    # Handle "max X" format
    if 'max' in duration:
        try:
            # Extract number after "max"
            max_match = re.search(r'max\s*(\d+)', duration)
            if max_match:
                return int(max_match.group(1))
        except:
            return None
            
    # Handle ranges like "35-40"
    if '-' in duration:
        try:
            low, high = map(int, re.findall(r'\d+', duration))
            return (low + high) // 2  # Use average for ranges
        except:
            return None
    
    # Try to extract just numbers
    try:
        return int(re.findall(r'\d+', duration)[0])
    except:
        return None

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

def plot_duration_distribution(df: pd.DataFrame) -> None:
    """
    Visualize the distribution of test durations, including max durations
    """
    if 'Duration' not in df.columns:
        st.warning("Duration column not found in dataset")
        return
        
    # Clean and convert durations
    durations = []
    max_durations = []
    invalid_durations = []
    special_values = {}
    
    for duration in df['Duration']:
        if pd.isna(duration):
            continue
            
        str_duration = str(duration).strip()
        cleaned_duration = clean_duration(str_duration)
        
        if cleaned_duration is not None:
            if 'max' in str_duration.lower():
                max_durations.append((cleaned_duration, str_duration))
            durations.append(cleaned_duration)
        else:
            # Collect special duration values for reporting
            special_value = str_duration.lower()
            if special_value:
                special_values[special_value] = special_values.get(special_value, 0) + 1
                invalid_durations.append(str_duration)
    
    if not durations:
        st.warning("No valid duration data found in dataset")
        return
        
    # Create histogram for duration distribution
    fig = px.histogram(
        x=durations,
        nbins=12,
        title='Distribution of Assessment Durations (Minutes)',
        labels={'x': 'Duration (minutes)', 'y': 'Number of Assessments'},
        color_discrete_sequence=['#36A2EB']
    )
    
    # Add mean and median lines
    mean_duration = sum(durations) / len(durations)
    median_duration = sorted(durations)[len(durations)//2]
    
    fig.add_vline(x=mean_duration, line_dash="dash", line_color="red",
                  annotation_text=f"Mean: {mean_duration:.1f} min")
    fig.add_vline(x=median_duration, line_dash="dash", line_color="green",
                  annotation_text=f"Median: {median_duration:.1f} min")
    
    # Add annotations for max durations
    for max_val, original_str in max_durations:
        fig.add_annotation(
            x=max_val,
            y=fig.data[0].y.max() * 0.9,  # Position at 90% of max height
            text=f"max {max_val}",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display statistics
    st.markdown("### Duration Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Duration", f"{mean_duration:.1f} min")
    with col2:
        st.metric("Median Duration", f"{median_duration:.1f} min")
    with col3:
        st.metric("Total Valid Samples", str(len(durations)))
        
    # Display non-numeric durations
    if invalid_durations:
        st.markdown("### Non-numeric Durations")
        
        # Count and display by category
        special_durations_count = Counter(special_values)
        special_durations_list = [f"{value} ({count})" for value, count in special_durations_count.most_common()]
        
        st.markdown("The following duration values were excluded from the analysis:")
        st.write(", ".join(special_durations_list))

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
