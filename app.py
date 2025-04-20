import streamlit as st
import pandas as pd
import time
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

st.set_page_config(
    page_title="SHL Assessment Recommender",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data(ttl=3600)
def load_metadata():
    return pd.read_csv("src/data/shl_full_catalog_with_duration_desc.csv")

df_meta = load_metadata()

params = st.query_params

@st.cache_resource
def load_model():
    try:
        from src.utils.model_evaluation import ModelEvaluator
        return ModelEvaluator("src/data/shl_full_catalog_with_duration_desc.csv")
    except Exception as e:
        st.error(f"Model initialization failed: {str(e)}")
        return None

if params.get("health", [""])[0] == "1":
    model = load_model()
    response = {
        "status": "healthy" if model is not None else "unhealthy",
        "model_status": "ready" if model is not None else "failed"
    }
    st.json(response)
    st.stop()

if params.get("api", [""])[0] == "1":
    model = load_model()
    
    if not model:
        st.json({
            "status": "error",
            "message": "Model failed to initialize",
            "recommended_assessments": []
        })
        st.stop()

    query = params.get("query", [""])[0]
    top_k = int(params.get("top_k", ["5"])[0])
    method = params.get("method", ["tfidf"])[0]

    try:
        result = model.evaluate_query(query, top_k=top_k, method=method)
        
        recommendations = []

        test_type_map = {
            "K": "Knowledge & Skills",
            "B": "Behavioral",
            "P": "Personality",
            "C": "Cognitive",
            "A": "Aptitude",
            "S": "Simulation",
            "E": "Emotional Intelligence",
            "D": "Development"
        }

        for item in result["results"][:top_k]:
            link = item.get("link", "")
            
            meta = df_meta[df_meta["Link"] == link]
            if meta.empty:
                continue
            row = meta.iloc[0]

            description = row.get("Description", "")
            duration_str = row.get("Duration", "Not")
            duration = int(duration_str.split()[0]) if duration_str.split()[0].isdigit() else 0

            raw_types = row.get("Test Types", "")
            test_types = [
                test_type_map.get(t.strip(), t.strip()) for t in raw_types.split(",") if t.strip()
            ]

            recommendations.append({
                "url": link,
                "adaptive_support": "Yes" if item.get("adaptiveIRTSupport", "").lower() == "yes" else "No",
                "description": description,
                "duration": duration,
                "remote_support": "Yes" if item.get("remoteTestingSupport", "").lower() == "yes" else "No",
                "test_type": test_types 
            })

        response = {
            "status": "success",
            "recommended_assessments": recommendations,
            "processing_time_ms": result["processing_time_ms"]
        }

        st.json(response)
        st.stop()
    except Exception as e:
        st.json({
            "status": "error",
            "message": str(e),
            "recommended_assessments": []
        })
        st.stop()

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

@st.cache_data(ttl=3600) 
def load_assessment_data():
    try:
        df = pd.read_csv('src/data/shl_full_catalog_with_duration_desc.csv')
        return df
    except Exception as e:
        st.error(f"Error loading assessment data: {str(e)}")
        return None

@st.cache_resource
def initialize_model(data_path: str):
    try:
        model = ModelEvaluator(data_path)
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

def get_recommendations(query: str, model: ModelEvaluator, method: str = 'hybrid', top_k: int = 5):
    try:
        start_time = time.time()
        results = model.evaluate_query(query, top_k=top_k, method=method)
        processing_time_ms = results['processing_time_ms']
        return results['results'], processing_time_ms
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        return [], 0

@st.cache_data
def generate_query_suggestions(df):
    job_roles = ["Data Scientist", "Software Engineer", "Business Analyst", "Project Manager", 
                "Team Lead", "Administrative Assistant", "Sales Manager", "Customer Support"]
    skills = ["programming", "leadership", "communication", "problem-solving", "technical", "analytical"]
    durations = ["under 30 minutes", "quick assessment", "comprehensive evaluation"]
    
    suggestions = []
    for role in job_roles:
        suggestions.append(f"{role} assessment")
        for skill in skills[:3]:  # Limit combinations
            suggestions.append(f"{skill} test for {role}")
    
    for skill in skills:
        for duration in durations[:2]:  # Limit combinations
            suggestions.append(f"{skill} assessment {duration}")
    
    return suggestions

def main():
    st.markdown('<div class="main-header">SHL Assessment Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Powered by AI to find the perfect assessment for your hiring needs</div>', unsafe_allow_html=True)
    
    df = load_assessment_data()
    if df is None:
        st.error("Failed to load assessment data. Please check the file path and try again.")
        return
    
    model = initialize_model('src/data/shl_full_catalog_with_duration_desc.csv')
    if model is None:
        st.error("Failed to initialize recommendation model. Please check the logs.")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs(["Recommender", "Catalog Analysis", "Model Evaluation", "About"])
    
    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        st.subheader("What assessment are you looking for?")
        
        query_suggestions = generate_query_suggestions(df)
        query = st.selectbox(
            "Enter your search query or select a suggestion",
            options=[""] + query_suggestions,
            index=0,
            help="Describe the job role, skills, or specific assessment you need"
        )
        
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
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
        
        with st.expander("Advanced Filters"):
            col1, col2 = st.columns(2)
            
            with col1:
                remote_only = st.checkbox("Remote Testing Support", value=False, 
                                        help="Only show assessments with remote testing capability")
                
            with col2:
                adaptive_only = st.checkbox("Adaptive Testing Support", value=False,
                                          help="Only show assessments with adaptive/IRT capability")
                
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
        
        submit_col1, submit_col2, submit_col3 = st.columns([1, 1, 1])
        with submit_col2:
            search_button = st.button("🔍 Find Assessments", type="primary", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if search_button and query:
            with st.spinner("Generating recommendations..."):
                start_time = time.time()
                recommendations, proc_time = get_recommendations(query, model, method=search_method, top_k=top_k)
                end_time = time.time()
                
                if recommendations:
                    recommendations = deduplicate_recommendations(recommendations)
                    
                    filtered_recommendations = recommendations
                    
                    if remote_only:
                        filtered_recommendations = [r for r in filtered_recommendations if r["remoteTestingSupport"].lower() == "yes"]
                    
                    if adaptive_only:
                        filtered_recommendations = [r for r in filtered_recommendations if r["adaptiveIRTSupport"].lower() == "yes"]
                    
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
    
    with tab2:
        st.subheader("SHL Assessment Catalog Analysis")
        
        st.markdown("### Test Type Distribution")
        st.write("This chart shows the distribution of different test types across the SHL assessment catalog.")
        plot_test_type_distribution(df)
        
        st.markdown("### Assessment Duration Analysis")
        plot_duration_distribution(df)
        
        st.markdown("### Testing Support Analysis")
        plot_remote_adaptive_support(df)
    
    with tab3:
        st.subheader("Model Evaluation")
        st.write("This section allows you to evaluate and compare different search methods using sample queries.")
        
        add_search_method_explanation()
        
        st.markdown("### Benchmark Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            benchmark_k = st.slider("Top-K for evaluation", 1, 10, 3, 
                                   help="Number of results to consider for metrics calculation")
        with col2:
            selected_metrics = st.multiselect("Metrics to calculate", 
                                            ["mean_recall_at_k", "map_at_k", "avg_processing_time_ms"],
                                            ["mean_recall_at_k", "map_at_k", "avg_processing_time_ms"])
        
        st.markdown("### Sample Benchmark Queries")
        benchmark_queries = get_sample_benchmark_queries()
        
        for i, query_item in enumerate(benchmark_queries):
            with st.expander(f"Query {i+1}: {query_item['query'][:60]}..."):
                st.write(query_item['query'])
                st.write("Relevant assessments:")
                for item in query_item['relevant_items']:
                    st.write(f"- {item}")
        
        if st.button("Run Benchmark"):
            with st.spinner("Running benchmark..."):
                queries = [item['query'] for item in benchmark_queries]
                ground_truth = {item['query']: item['relevant_items'] for item in benchmark_queries}
                
                benchmark_results = run_benchmark(
                    model, 
                    queries, 
                    ground_truth,
                    methods=['semantic', 'tfidf', 'hybrid'],
                    top_k=benchmark_k
                )
                
                st.markdown("### Benchmark Results")
                st.dataframe(benchmark_results)
                
                for metric in selected_metrics:
                    if metric in benchmark_results.columns:
                        plot_benchmark_comparison(benchmark_results, metric)
                
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
    
    with tab4:
        st.markdown("""
        ## About SHL Assessment Recommender
                
        ### Key Features
        
        - 🔍 **Hybrid Search Methodology** combining semantic and keyword-based approaches  
        - 📊 **Advanced Recommendation Engine** with configurable search methods  
        - ⚡ **Real-time Assessment Matching**  
        - 📈 **Comprehensive Filtering Options**  
        """)

        st.markdown("### Core Technologies")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            - Python (Core language)  
            - FastAPI (REST API framework)  
            - Streamlit (Web app framework)  
            """)

        with col2:
            st.markdown("""
            - SentenceTransformer (Neural embeddings)  
            - Scikit-learn (TF-IDF & ML utilities)  
            - Pandas & NumPy (Data wrangling)  
            """)

        with col3:
            st.markdown("""
            - BeautifulSoup (Web scraping)  
            - Plotly & Matplotlib (Visualization)  
            """)

        st.markdown("""
        ### Recommendation Process
        
        1. Load pre-processed SHL assessment catalog  
        2. Generate semantic and keyword representations  
        3. Match queries using hybrid search techniques  
        4. Rank and return most relevant assessments  

        ### API Endpoints

        1. **Assessment Recommendations**
        ```
        https://shl-assessment-9vuzh8xdu2vvchdhvcnxtq.streamlit.app/?api=1&query=java+developer+40+minutes&top_k=5
        ```
        - Main endpoint for getting assessment recommendations  
        - Accepts job descriptions and skill-based queries  
        - Configurable number of recommendations (`top_k` parameter)  
        - Returns detailed assessment information as a JSON output  

        2. **Health Status**
        ```
        https://shl-assessment-9vuzh8xdu2vvchdhvcnxtq.streamlit.app/?health=1
        ```
        - Monitors API availability and status  
        - Returns real-time health information  
        - Used for service monitoring  

        ### Source Code
        Access the complete project on GitHub: [SHL Recommender](https://github.com/Devam153/SHL-assessment)
        """)
    
    st.markdown("""
    <div class="footer">
        SHL Assessment Recommender • Built with Streamlit • Data Source: SHL Product Catalog
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
