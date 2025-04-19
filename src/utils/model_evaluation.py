import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self, 
        data_path: str,
        model_name: str = 'paraphrase-MiniLM-L6-v2',
        cache_dir: str = None
    ):
        """Initialize the model evaluator with dataset and model configurations."""
        self.data_path = Path(data_path)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Load and preprocess data
        self._load_data()
        
        # Initialize models
        try:
            logger.info(f"Initializing transformer model: {model_name}")
            self.transformer = SentenceTransformer(model_name)
            logger.info(f"Initializing TF-IDF vectorizer")
            self.tfidf = TfidfVectorizer(stop_words='english')
            
            # Compute embeddings
            self._compute_embeddings()
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
        
    def _load_data(self) -> None:
        """Load and preprocess the SHL assessment catalog."""
        try:
            logger.info(f"Loading assessment catalog from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} assessments")
            
            # Map original column names to internal names to handle different formats
            column_mapping = {
                'Test Name': 'testName',
                'Link': 'link',
                'Remote Testing': 'remoteTestingSupport',
                'Adaptive/IRT': 'adaptiveIRTSupport',
                'Test Types': 'testTypes',
                'Duration': 'duration'
            }
            
            # Check if columns exist and rename accordingly
            for orig, new in column_mapping.items():
                if orig in self.df.columns:
                    self.df = self.df.rename(columns={orig: new})
            
            # Validate required columns after renaming
            required_cols = ['testName', 'link', 'remoteTestingSupport', 'adaptiveIRTSupport']
            for col in required_cols:
                if col not in self.df.columns:
                    logger.error(f"Required column '{col}' not found in dataset after renaming.")
                    raise ValueError(f"Required column '{col}' not found in dataset.")
            
            # Add default fields if missing
            if 'testTypes' not in self.df.columns:
                logger.warning("Test Types column is missing. Adding default value.")
                self.df['testTypes'] = "Not specified"
                
            if 'duration' not in self.df.columns:
                logger.warning("Duration column is missing. Adding default value.")
                self.df['duration'] = "Not specified"
            
            # Create combined description field for better matching
            self.df['combined_description'] = (
                self.df['testName'] + ' ' + 
                self.df['testTypes'] + ' ' + 
                self.df['remoteTestingSupport'] + ' ' + 
                self.df['adaptiveIRTSupport']
            ).fillna('')
            
            logger.info("Assessment data loaded and preprocessed successfully")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
        
    def _compute_embeddings(self) -> None:
        """Compute embeddings for all assessments."""
        try:
            cache_file = self.cache_dir / f"embeddings_{self.model_name}.npy" if self.cache_dir else None
            
            if cache_file and cache_file.exists():
                logger.info("Loading cached embeddings...")
                self.embeddings = np.load(cache_file)
                logger.info(f"Loaded embeddings with shape {self.embeddings.shape}")
            else:
                logger.info("Computing embeddings...")
                descriptions = self.df['combined_description'].tolist()
                logger.info(f"Computing embeddings for {len(descriptions)} descriptions")
                
                self.embeddings = self.transformer.encode(
                    descriptions,
                    show_progress_bar=True
                )
                
                if cache_file:
                    logger.info(f"Caching embeddings to {cache_file}")
                    # Ensure directory exists
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache_file, self.embeddings)
            
            # Compute TF-IDF matrix
            logger.info("Computing TF-IDF matrix")
            self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_description'])
            logger.info(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            raise
        
    def _semantic_search(self, query: str, top_k: int = 5) -> list:
        """Perform semantic search using transformer embeddings."""
        query_embedding = self.transformer.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'testName': self.df.iloc[idx]['testName'],
                'score': float(similarities[idx]),
                'index': int(idx),
                'testTypes': self.df.iloc[idx]['testTypes'],
                'duration': self.df.iloc[idx]['duration'],
                'remoteTestingSupport': self.df.iloc[idx]['remoteTestingSupport'],
                'adaptiveIRTSupport': self.df.iloc[idx]['adaptiveIRTSupport'],
                'link': self.df.iloc[idx]['link']
            }
            for idx in top_indices
        ]
    
    def _tfidf_search(self, query: str, top_k: int = 5) -> list:
        """Perform TF-IDF based search."""
        query_vec = self.tfidf.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {
                'testName': self.df.iloc[idx]['testName'],
                'score': float(similarities[idx]),
                'index': int(idx),
                'testTypes': self.df.iloc[idx]['testTypes'],
                'duration': self.df.iloc[idx]['duration'],
                'remoteTestingSupport': self.df.iloc[idx]['remoteTestingSupport'],
                'adaptiveIRTSupport': self.df.iloc[idx]['adaptiveIRTSupport'],
                'link': self.df.iloc[idx]['link']
            }
            for idx in top_indices
        ]
    
    def _hybrid_search(self, query: str, top_k: int = 5, alpha: float = 0.7) -> list:
        """Perform hybrid search combining semantic and TF-IDF approaches."""
        # Get semantic similarities
        query_embedding = self.transformer.encode([query])[0]
        sem_similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get TF-IDF similarities
        query_vec = self.tfidf.transform([query])
        tfidf_similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Combine scores
        combined_similarities = alpha * sem_similarities + (1 - alpha) * tfidf_similarities
        top_indices = np.argsort(combined_similarities)[-top_k:][::-1]
        
        return [
            {
                'testName': self.df.iloc[idx]['testName'],
                'score': float(combined_similarities[idx]),
                'index': int(idx),
                'testTypes': self.df.iloc[idx]['testTypes'],
                'duration': self.df.iloc[idx]['duration'],
                'remoteTestingSupport': self.df.iloc[idx]['remoteTestingSupport'],
                'adaptiveIRTSupport': self.df.iloc[idx]['adaptiveIRTSupport'],
                'link': self.df.iloc[idx]['link']
            }
            for idx in top_indices
        ]
    
    def evaluate_query(
        self,
        query: str,
        top_k: int = 5,
        method: str = 'hybrid',
        alpha: float = 0.7
    ) -> dict:
        """Evaluate a single query using specified method."""
        start_time = time.time()
        
        try:
            if method == 'semantic':
                results = self._semantic_search(query, top_k)
            elif method == 'tfidf':
                results = self._tfidf_search(query, top_k)
            else:  # hybrid
                results = self._hybrid_search(query, top_k, alpha)
                
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'results': results,
                'processing_time_ms': processing_time,
                'method': method
            }
        except Exception as e:
            logger.error(f"Error evaluating query '{query}' with method '{method}': {str(e)}")
            return {
                'results': [],
                'processing_time_ms': 0,
                'method': method,
                'error': str(e)
            }
    
    def benchmark_queries(
        self,
        queries: list,
        methods: list = ['semantic', 'tfidf', 'hybrid'],
        top_k: int = 5
    ) -> pd.DataFrame:
        """Benchmark multiple queries across different methods."""
        results = []
        
        for query in queries:
            for method in methods:
                eval_result = self.evaluate_query(query, top_k, method)
                results.append({
                    'query': query,
                    'method': method,
                    'processing_time_ms': eval_result['processing_time_ms'],
                    'top_results': [r['testName'] for r in eval_result['results']],
                    'relevance_scores': [r['score'] for r in eval_result['results']]
                })
        
        return pd.DataFrame(results)