# SHL Assessment Recommender 

Project Website: https://shl-assessment-9vuzh8xdu2vvchdhvcnxtq.streamlit.app

API endpoint (query) - https://shl-assessment-zlp2.onrender.com/recommend?query=java+developer+40+minutes&top_k=5


API endpoint (health check) - https://shl-assessment-zlp2.onrender.com/health

## Overview

This document outlines our approach to building the SHL Assessment Recommendation System, which matches job descriptions or natural language queries with relevant SHL assessments. The system employs a hybrid search methodology combining semantic understanding and keyword matching.

## Architecture

The solution consists of two main components:

- A **Streamlit web application** for interactive exploration and visualization
- A **FastAPI REST API** for programmatic access and integration which is deployed separately on render

## Data Processing

### Data Collection & Enrichment

1. Started with the base SHL product catalog
2. Enhanced the dataset by:
   - **Web scraping** assessment duration information using **BeautifulSoup**
   - Converting raw test type codes to user-friendly descriptions
   - Combining fields to create rich text representations for search

### Text Processing Pipeline

- **Tokenization** and **lemmatization** for standardized text representation
- **Combined feature representation** using assessment name, test types, and support features
- **Vector embeddings** generation for semantic similarity calculations

## Search Methodologies

We implemented three complementary search approaches:

1. **Semantic Search**:

   - Leverages **SentenceTransformer** (paraphrase-MiniLM-L6-v2) for dense vector embeddings
   - Captures meaning and context beyond keywords
   - Optimized for finding conceptually related assessments

2. **TF-IDF Search**:

   - Implements **TF-IDF vectorization** using scikit-learn
   - Focuses on keyword frequency and importance
   - Better at exact term matching and specialized terminology

3. **Hybrid Search**:
   - Combines semantic and TF-IDF scores with configurable weighting
   - Balances conceptual understanding with keyword precision
   - Default approach with performance superior to either method alone

## Implementation Details

### Backend (Python)

- **FastAPI** for high-performance, standards-compliant REST API
- **Model Evaluation** module for handling different search methodologies
- **Sentence Transformers** for generating semantic embeddings
- **NumPy** and **scikit-learn** for efficient vector operations
- **Pandas** for data manipulation and transformation

### Web Application (Streamlit)

- Interactive **Streamlit** dashboard with multiple visualization tabs
- **Plotly** and **Matplotlib** for interactive data visualizations
- Real-time recommendation display with detailed assessment information
- Advanced filtering options by test types, duration, and support features

### Benchmarking & Evaluation

- Comprehensive model evaluation using **Mean Recall@K** and **MAP@K**
- Performance comparison across search methodologies
- Visual benchmarking tools for analyzing search quality

## API Design

The API provides two primary endpoints:

1. `/health` - Service health check returning HTTP 200 when healthy
2. `/recommend` - Main recommendation endpoint accepting:
   - `query`: Job description or natural language query (required)
   - `top_k`: Number of recommendations to return (1-10) (optional)

Response format includes recommended assessments with:

- URL
- Test name
- Duration
- Remote and adaptive testing support
- Test types

## Performance Optimizations

- **Embedding caching** for faster response times
- **Parallel processing** of multiple search methods
- **Query expansion** techniques for improved recall
- **Score normalization** for consistent ranking

## Features & Capabilities

- **Natural language understanding** of job requirements and skills
- **Test type filtering** based on specific assessment needs
- **Duration constraints** processing (e.g., "tests under 30 minutes")
- **Remote testing** and **adaptive testing** support filtering
- **Interactive visualizations** of the assessment catalog
- **Customizable search parameters** for different use cases

## Technologies Used

- **Python 3.9+** - Core programming language
- **FastAPI** - REST API framework
- **Streamlit** - Web application framework
- **SentenceTransformer** - Neural embedding model
- **scikit-learn** - ML utilities and TF-IDF vectorization
- **Pandas & NumPy** - Data processing
- **BeautifulSoup** - Web scraping for dataset enrichment
- **Plotly & Matplotlib** - Data visualization
- **Render** - API deployment platform

## Deployment

The application is deployed on the Streamlit Community Cloud

The API is deployed on Render

## Conclusion

The SHL Assessment Recommender demonstrates the power of combining traditional information retrieval techniques with modern NLP approaches. The hybrid search methodology delivers superior results by leveraging both semantic understanding and keyword precision. The system architecture provides both user-friendly interactive exploration and standardized programmatic access.
