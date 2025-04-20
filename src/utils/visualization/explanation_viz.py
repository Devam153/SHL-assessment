
"""
Visualization utilities for explanations and documentation
"""
import streamlit as st

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
            - Uses **term frequency-inverse document frequency** technique
            - Good at matching specific keywords and technical terms
            - Works well when exact terminology is important
            - Based on statistical word occurrence patterns
            """)
            
        with col3:
            st.markdown("### Hybrid Search")
            st.markdown("""
            - Combines **both semantic and TF-IDF approaches**
            - Balances meaning-based and keyword-based matching
            - Often provides the best overall performance
            - Default recommendation method in the system
            """)
            
        st.info("The benchmark results show how each method performs on sample queries. Higher recall and precision values indicate better performance, while lower processing time indicates faster results.")
