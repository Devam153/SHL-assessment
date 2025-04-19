
import streamlit as st
import pandas as pd
import numpy as np
import time
from typing import Dict, List, Any, Optional
from src.utils.model_evaluation import ModelEvaluator
from src.utils.visualization import (
    plot_benchmark_comparison, 
    plot_test_type_distribution,
    plot_remote_adaptive_support,
    display_recommendation_details,
    add_search_method_explanation,
    plot_duration_distribution,
    deduplicate_recommendations
)
from src.utils.benchmark import get_sample_benchmark_queries, run_benchmark
from src.utils.api_utils import format_api_response, health_check, validate_query_params

# Set page configuration
st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1️⃣ grab raw query params via the new API
params = st.query_params

# 2️⃣ detect “api mode”
if params.get("api", [""])[0] == "1":
    # parse parameters
    query  = params.get("query", [""])[0]
    top_k  = int(params.get("top_k", ["5"])[0])
    method = params.get("method", ["hybrid"])[0]

    # load & run your model
    from src.utils.model_evaluation import ModelEvaluator
    model = ModelEvaluator("src/data/shl_full_catalog_with_duration.csv")
    result = model.evaluate_query(query, top_k=top_k, method=method)

    # build JSON response
    recs = []
    for item in result["results"]:
        recs.append({
            "url":              item["link"],
            "adaptive_support": item["adaptiveIRTSupport"],
            "description":      item["testName"],
            "duration":         item["duration"],
            "remote_support":   item["remoteTestingSupport"],
            "test_type":        [c.strip() for c in item["testTypes"].split(",")]
        })

    out = {
        "status":             "success",
        "recommendations":    recs[:top_k],
        "processing_time_ms": result["processing_time_ms"]
    }

    # emit JSON & halt the rest of the app
    st.json(out)
    st.stop()

# Apply custom styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1rem;
    }
    .highlight {
        background-color: #DBEAFE;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
    }
    .footer {
        margin-top: 3rem;
        color: #6B7280;
        font-size: 0.8rem;
        text-align: center;
    }
    .plot-container {
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1.5rem;
        padding: 1rem;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading to improve performance
@st.cache_data(ttl=3600)  # cache for 1 hour
def load_assessment_data():
    try:
        df = pd.read_csv('src/data/shl_full_catalog_with_duration.csv')
        return df
    except Exception as e:
        st.error(f"Error loading assessment data: {str(e)}")
        return None

# Cache model initialization
@st.cache_resource
def initialize_model(data_path: str):
    try:
        model = ModelEvaluator(data_path)
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# Get recommendations 
def get_recommendations(query: str, model: ModelEvaluator, method: str = 'hybrid', top_k: int = 5):
    try:
        start_time = time.time()
        results = model.evaluate_query(query, top_k=top_k, method=method)
        processing_time_ms = results['processing_time_ms']
        return results['results'], processing_time_ms
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return [], 0

# Advanced query suggestions based on assessment data
@st.cache_data
def generate_query_suggestions(df):
    job_roles = ["Data Scientist", "Software Engineer", "Business Analyst", "Project Manager", 
                "Team Lead", "Administrative Assistant", "Sales Manager", "Customer Support"]
    skills = ["programming", "leadership", "communication", "problem-solving", "technical", "analytical"]
    durations = ["under 30 minutes", "quick assessment", "comprehensive evaluation"]
    
    # Generate combinations
    suggestions = []
    for role in job_roles:
        suggestions.append(f"{role} assessment")
        for skill in skills[:3]:  # Limit combinations
            suggestions.append(f"{skill} test for {role}")
    
    for skill in skills:
        for duration in durations[:2]:  # Limit combinations
            suggestions.append(f"{skill} assessment {duration}")
    
    return suggestions

# Main application
def main():
    # Header
    st.markdown('<div class="main-header">SHL Assessment Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by AI to find the perfect assessment for your hiring needs</div>', unsafe_allow_html=True)
    
    # Load data
    df = load_assessment_data()
    if df is None:
        st.error("Failed to load assessment data. Please check the file path and try again.")
        return
    
    # Initialize model
    model = initialize_model('src/data/shl_full_catalog_with_duration.csv')
    if model is None:
        st.error("Failed to initialize recommendation model. Please check the logs.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Recommender", "Catalog Analysis", "Model Evaluation", "About"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Input form
        st.subheader("What assessment are you looking for?")
        
        # Query input with suggestions
        query_suggestions = generate_query_suggestions(df)
        query = st.selectbox(
            "Enter your search query or select a suggestion",
            options=[""] + query_suggestions,
            index=0,
            help="Describe the job role, skills, or specific assessment you need"
        )
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            # User can also type a custom query
            custom_query = st.text_input(
                "Or type your own custom query:",
                placeholder="e.g., Leadership assessment for mid-level managers with focus on decision making",
                help="Be specific about the role, skills, and requirements"
            )
            
            if custom_query:
                query = custom_query
        
        with col2:
            top_k = st.slider(
                "Number of results",
                min_value=1,
                max_value=10,
                value=5,
                help="How many assessment recommendations do you want to see?"
            )
        
        with col3:
            search_method = st.selectbox(
                "Search method",
                options=["hybrid", "semantic", "tfidf"],
                index=0,
                help="Choose the search algorithm to use"
            )
        
        # Advanced filters (collapsible)
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                remote_only = st.checkbox("Remote Testing Support", value=False, 
                                        help="Only show assessments with remote testing capability")
                
            with col2:
                adaptive_only = st.checkbox("Adaptive Testing Support", value=False,
                                          help="Only show assessments with adaptive/IRT capability")
                
            # Filter by test types
            st.write("Test Types (select at least one)")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                type_a = st.checkbox("Aptitude (A)", value=True)
                type_b = st.checkbox("Behavioral (B)", value=True)
                type_c = st.checkbox("Cognitive (C)", value=True)
            
            with col2:
                type_p = st.checkbox("Personality (P)", value=True)
                type_s = st.checkbox("Situational (S)", value=True)
                type_t = st.checkbox("Technical (T)", value=True)
                
            with col3:
                type_k = st.checkbox("Knowledge (K)", value=True)
                type_l = st.checkbox("Leadership (L)", value=True)
                type_d = st.checkbox("Decision Making (D)", value=True)
                
            with col4:
                type_n = st.checkbox("Numerical (N)", value=True)
                type_e = st.checkbox("Emotional Intelligence (E)", value=True)
        
        # Submit button
        submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 1])
        with submit_col2:
            search_button = st.button("🔍 Find Assessments", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Get recommendations when button is clicked
        if search_button and query:
            with st.spinner("Generating recommendations..."):
                start_time = time.time()
                recommendations, proc_time = get_recommendations(query, model, method=search_method, top_k=top_k)
                end_time = time.time()
                
                if recommendations:
                    # Deduplicate recommendations first
                    recommendations = deduplicate_recommendations(recommendations)
                    
                    # Apply advanced filters if set
                    filtered_recommendations = recommendations
                    
                    if remote_only:
                        filtered_recommendations = [r for r in filtered_recommendations if r["remoteTestingSupport"].lower() == "yes"]
                    
                    if adaptive_only:
                        filtered_recommendations = [r for r in filtered_recommendations if r["adaptiveIRTSupport"].lower() == "yes"]
                    
                    # Filter by test types
                    selected_types = []
                    if type_a: selected_types.append("A")
                    if type_b: selected_types.append("B") 
                    if type_c: selected_types.append("C")
                    if type_p: selected_types.append("P")
                    if type_s: selected_types.append("S")
                    if type_t: selected_types.append("T")
                    if type_k: selected_types.append("K")
                    if type_l: selected_types.append("L")
                    if type_d: selected_types.append("D")
                    if type_n: selected_types.append("N")
                    if type_e: selected_types.append("E")
                    
                    if selected_types:
                        filtered_recommendations = [
                            r for r in filtered_recommendations 
                            if any(t in selected_types for t in r["testTypes"].replace(" ", "").split(","))
                        ]
                    
                    # Display results using the visualization module
                    if filtered_recommendations:
                        st.success(f"Found {len(filtered_recommendations)} matching assessments in {proc_time:.2f} ms")
                        for recommendation in filtered_recommendations:
                            display_recommendation_details(recommendation)
                    else:
                        st.warning("No assessments match your filters. Try adjusting your criteria.")
                else:
                    st.info("No matching assessments found. Try a different query.")
        elif search_button and not query:
            st.warning("Please enter a search query or select a suggestion.")
    
    # Catalog Analysis tab
    with tab2:
        st.subheader("SHL Assessment Catalog Analysis")
        
        st.markdown("### Test Type Distribution")
        st.write("This chart shows the distribution of different test types across the SHL assessment catalog.")
        plot_test_type_distribution(df)
        
        st.markdown("### Assessment Duration Analysis")
        plot_duration_distribution(df)
        
        st.markdown("### Testing Support Analysis")
        plot_remote_adaptive_support(df)
    
    # Model Evaluation tab
    with tab3:
        st.subheader("Model Evaluation")
        st.write("This section allows you to evaluate and compare different search methods using sample queries.")
        
        # Add explanation about search methods
        add_search_method_explanation()
        
        # Benchmark controls
        st.markdown("### Benchmark Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            benchmark_k = st.slider("Top-K for evaluation", 1, 10, 3, 
                                   help="Number of results to consider for metrics calculation")
        with col2:
            selected_metrics = st.multiselect("Metrics to calculate", 
                                            ["mean_recall_at_k", "map_at_k", "avg_processing_time_ms"],
                                            ["mean_recall_at_k", "map_at_k", "avg_processing_time_ms"])
        
        # Sample benchmark queries
        st.markdown("### Sample Benchmark Queries")
        benchmark_queries = get_sample_benchmark_queries()
        
        # Display sample queries
        for i, query_item in enumerate(benchmark_queries):
            with st.expander(f"Query {i+1}: {query_item['query'][:60]}..."):
                st.write(query_item['query'])
                st.write("Relevant assessments:")
                for item in query_item['relevant_items']:
                    st.write(f"- {item}")
        
        # Run benchmark button
        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark..."):
                # Extract queries and ground truth
                queries = [item['query'] for item in benchmark_queries]
                ground_truth = {item['query']: item['relevant_items'] for item in benchmark_queries}
                
                # Run benchmark
                benchmark_results = run_benchmark(
                    model, 
                    queries, 
                    ground_truth,
                    methods=['semantic', 'tfidf', 'hybrid'],
                    top_k=benchmark_k
                )
                
                # Display results
                st.markdown("### Benchmark Results")
                st.dataframe(benchmark_results)
                
                # Plot results
                for metric in selected_metrics:
                    if metric in benchmark_results.columns:
                        plot_benchmark_comparison(benchmark_results, metric)
                
                # Add explanation of results
                with st.expander("💡 Interpreting Benchmark Results"):
                    st.markdown("""
                    ### Understanding the Metrics
                    
                    - **Mean Recall@K**: Measures what percentage of the relevant assessments were successfully found in the top K results. Higher is better.
                    - **Mean Average Precision@K**: Considers both the presence of relevant items and their ranking position. Higher is better.
                    - **Avg Processing Time**: How long it takes to process a query in milliseconds. Lower is better.
                    
                    ### About the Search Methods
                    
                    - **Semantic**: Uses neural embeddings to understand meaning and context.
                    - **TF-IDF**: Uses keyword frequency statistics for matching.
                    - **Hybrid**: Combines both approaches for balanced results.
                    
                    If you're seeing zeros across all methods, the system might be unable to find matches between the predictions and ground truth. This could happen if:
                    
                    1. The assessment names in the catalog don't match the ground truth items exactly
                    2. The search methods need further tuning for your specific use case
                    3. The dataset needs to be preprocessed differently
                    """)
    
    # About tab
    with tab4:
        st.markdown("""
        ## About SHL Assessment Recommender
        
        This tool uses artificial intelligence to help HR professionals and hiring managers find the most suitable SHL assessments for their candidate evaluation needs.
        
        ### Features
        
        - 🔍 **Advanced search methods** including semantic search, TF-IDF, and hybrid approaches
        - 📊 **Comprehensive evaluation metrics** (Mean Recall@K, MAP@K)
        - ⚡ **Fast, real-time recommendations** with efficient caching
        - 📈 **Detailed analytics** on the assessment catalog
        - 🌐 **REST API** for integration with other systems
        
        ### Technologies Used
        
        - **Sentence Transformers**: For generating semantic embeddings of text
        - **Scikit-learn**: For TF-IDF vectorization and similarity calculations
        - **Streamlit**: For creating the interactive web interface
        - **Pandas & NumPy**: For data manipulation and analysis
        - **Plotly**: For interactive data visualizations
        
        ### How It Works
        
        1. The system loads the SHL assessment catalog and pre-computes embeddings for each assessment
        2. When you enter a query, both semantic and TF-IDF representations are generated
        3. The system finds assessments with similar representations to your query
        4. Results are combined, ranked by relevance, and returned
        
        ### API Documentation
        
        The system exposes a REST API for programmatic access. The main endpoints are:
        
        - `GET /api/recommendations?query=your_query&top_k=5`: Get assessment recommendations
        - `GET /api/health`: Check API health status
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        SHL Assessment Recommender • Built with Streamlit • Data Source: SHL Product Catalog
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
