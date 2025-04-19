# SHL Assessment Recommender

An intelligent recommendation system for SHL assessments based on job descriptions and natural language queries.

## Features

- **Advanced Search**: Uses semantic search, TF-IDF, and hybrid approaches to find relevant assessments
- **Interactive UI**: Built with Streamlit for easy exploration and visualization
- **REST API**: Provides programmatic access to recommendations
- **Performance Metrics**: Evaluation using Mean Recall@K and MAP@K
- **Analytics**: Visualizations of the assessment catalog data

## Project Structure

```
├── app.py                  # Streamlit web application
├── api.py                  # FastAPI REST API
├── src/
│   ├── data/               # Data files
│   │   └── shl_full_catalog.csv
│   ├── utils/              # Utility functions
│   │   ├── model_evaluation.py
│   │   ├── api_utils.py
│   │   ├── benchmark_utils.py
│   │   └── visualization.py
│   └── types/              # Type definitions
│       └── types.ts
└── tests/                  # Unit tests
    └── test_model_evaluation.py
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the web application:
   ```
   streamlit run app.py
   ```
4. Run the API:
   ```
   uvicorn api:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `GET /api/health`: Check API health status
- `GET /api/recommendations?query={query}&top_k={k}`: Get assessment recommendations
- `GET /api/methods`: Get information about available search methods

## Evaluation Metrics

The system uses the following metrics to evaluate recommendation quality:

- **Mean Recall@K**: Measures how many relevant assessments are retrieved in the top K recommendations
- **MAP@K (Mean Average Precision)**: Evaluates both relevance and ranking order of retrieved assessments

## Implementation Approach

1. **Data Preprocessing**:

   - Extract assessment metadata from SHL's product catalog
   - Create combined text representations for each assessment

2. **Text Representation**:

   - Semantic embeddings using Sentence Transformers
   - TF-IDF vectors for keyword-based matching

3. **Search Methods**:

   - Semantic search using cosine similarity
   - TF-IDF search for keyword matching
   - Hybrid search combining both approaches

4. **Ranking & Filtering**:
   - Score normalization and combination
   - Post-filtering based on user requirements
   - Comprehensive evaluation using benchmark queries

## License

MIT
