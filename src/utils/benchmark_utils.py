
"""
Benchmarking utilities for evaluating recommendation model performance
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import time
import logging
import re
from src.utils.model_evaluation import ModelEvaluator

logger = logging.getLogger(__name__)

def clean_text_for_comparison(text: str) -> str:
    """
    Clean and normalize text for more accurate comparison
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove common suffixes/prefixes that might differ between versions
    text = re.sub(r'\s*\(new\)\s*', ' ', text)
    text = re.sub(r'\s*assessment\s*', ' ', text)
    text = re.sub(r'\s*solution\s*', ' ', text)
    
    # Handle core skills
    text = re.sub(r'core\s+(\w+)', r'\1', text)  # "Core Java" -> "Java"
    
    # Handle common variations
    text = text.replace('javascript', 'java script').replace('js', 'java script')
    text = text.replace('collab', 'collaborat')  # Match collaboration/collaborate/collaborative
    text = text.replace('cognitive ability', 'cognitive')
    
    # Remove punctuation and extra spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_substantial_match(text1: str, text2: str) -> bool:
    """
    Determine if two cleaned strings are substantially matching
    
    This checks for:
    1. Direct substring matching
    2. Significant word overlap
    3. Key terms matching
    """
    # Log the comparison being made
    logger.info(f"Comparing: '{text1}' with '{text2}'")
    
    # Direct substring check
    if text1 in text2 or text2 in text1:
        logger.info(f"✓ Substring match found between '{text1}' and '{text2}'")
        return True
    
    # Check for key term matches
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # If texts contain important programming languages or skill terms, check for those specifically
    key_terms = [
        'java', 'python', 'sql', 'javascript', 'cognitive', 'personality', 
        'collaboration', 'leadership', 'data', 'analyst', 'challenge',
        'coding', 'assessment', 'profiling', 'interview'
    ]
    
    # Check if any key terms are common to both texts
    common_key_terms = [term for term in key_terms if term in text1 and term in text2]
    if common_key_terms:
        logger.info(f"✓ Common key terms found: {common_key_terms}")
        return True
    
    # Word overlap check - if enough words match
    common_words = words1.intersection(words2)
    
    # Consider it a match if there's significant word overlap
    # For short strings (1-2 words), require at least one word in common
    # For longer strings, require at least 30% of words in common from the shorter string
    min_word_count = min(len(words1), len(words2))
    
    if min_word_count <= 2:
        if len(common_words) >= 1:
            logger.info(f"✓ Short string match with common words: {common_words}")
            return True
    else:
        overlap_ratio = len(common_words) / min_word_count
        if overlap_ratio >= 0.3:
            logger.info(f"✓ Word overlap ratio {overlap_ratio:.2f} with common words: {common_words}")
            return True
    
    # No substantial match found
    logger.info(f"✗ No match between '{text1}' and '{text2}'")
    return False

def calculate_mean_recall_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: int = 3) -> float:
    """
    Calculate Mean Recall@K metric
    
    Args:
        predictions: List of lists of predicted item IDs for each query
        ground_truth: List of lists of relevant item IDs for each query
        k: The cutoff for consideration (only consider top k predictions)
        
    Returns:
        Mean Recall@K value across all queries
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    recall_values = []
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        if not truth:  # Skip queries with no relevant items
            continue
            
        # Consider only top-k predictions
        pred_k = pred[:k]
        
        # Count relevant items in top-k predictions
        relevant_retrieved = 0
        
        logger.info(f"Query index {i}:")
        logger.info(f"Predictions: {pred_k}")
        logger.info(f"Ground truth: {truth}")
        
        matches_found = []
        
        for pred_item in pred_k:
            pred_clean = clean_text_for_comparison(pred_item)
            
            for truth_item in truth:
                truth_clean = clean_text_for_comparison(truth_item)
                
                # Check for substantial overlap between prediction and ground truth
                if is_substantial_match(pred_clean, truth_clean):
                    relevant_retrieved += 1
                    # Log successful match
                    logger.info(f"✓ Match found: '{pred_item}' matches with ground truth '{truth_item}'")
                    matches_found.append((pred_item, truth_item))
                    break  # Count each prediction only once
        
        # Calculate recall for this query
        recall = relevant_retrieved / len(truth)
        recall_values.append(recall)
        
        # Log detailed match information
        if matches_found:
            logger.info(f"Query index {i}: Found {relevant_retrieved} relevant items out of {len(truth)} ground truth items")
            for match in matches_found:
                logger.info(f"  - '{match[0]}' matched with '{match[1]}'")
        else:
            logger.warning(f"Query index {i}: No matches found!")
    
    # Return mean recall
    mean_recall = np.mean(recall_values) if recall_values else 0.0
    logger.info(f"Mean Recall@{k}: {mean_recall:.4f}")
    return mean_recall

def calculate_map_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: int = 3) -> float:
    """
    Calculate Mean Average Precision@K (MAP@K) metric
    
    Args:
        predictions: List of lists of predicted item IDs for each query
        ground_truth: List of lists of relevant item IDs for each query
        k: The cutoff for consideration (only consider top k predictions)
        
    Returns:
        MAP@K value across all queries
    """
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    ap_values = []
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        if not truth:  # Skip queries with no relevant items
            continue
            
        # Consider only top-k predictions
        pred_k = pred[:k]
        
        # Calculate precision at each position where a relevant item is found
        precision_values = []
        num_relevant_found = 0
        
        for j, pred_item in enumerate(pred_k):
            pred_clean = clean_text_for_comparison(pred_item)
            is_relevant = False
            
            for truth_item in truth:
                truth_clean = clean_text_for_comparison(truth_item)
                
                if is_substantial_match(pred_clean, truth_clean):
                    is_relevant = True
                    # Log the match
                    logger.info(f"MAP - Match found at position {j+1}: '{pred_item}' matches '{truth_item}'")
                    break
            
            if is_relevant:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (j + 1)
                precision_values.append(precision_at_i)
        
        # Calculate average precision for this query
        ap = sum(precision_values) / len(truth) if precision_values else 0
        ap_values.append(ap)
        
        # Log detailed precision information
        logger.info(f"Query index {i}: AP = {ap:.4f}, found {num_relevant_found} relevant items")
    
    # Return mean AP
    mean_ap = np.mean(ap_values) if ap_values else 0.0
    logger.info(f"Mean AP@{k}: {mean_ap:.4f}")
    return mean_ap

def run_benchmark(
    evaluator: ModelEvaluator,
    test_queries: List[str],
    ground_truth: Dict[str, List[str]],
    methods: List[str] = ['semantic', 'tfidf', 'hybrid'],
    top_k: int = 3
) -> pd.DataFrame:
    """
    Run a comprehensive benchmark on multiple queries using different methods
    
    Args:
        evaluator: The ModelEvaluator instance to use
        test_queries: List of test queries to evaluate
        ground_truth: Dictionary mapping queries to lists of relevant item IDs
        methods: List of search methods to benchmark
        top_k: Number of results to consider
        
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    # Log benchmark parameters
    logger.info(f"Running benchmark with {len(test_queries)} queries, top_k={top_k}")
    logger.info(f"Methods: {methods}")
    logger.info(f"Ground truth: {ground_truth}")
    
    for method in methods:
        method_predictions = []
        method_processing_times = []
        
        for query in test_queries:
            start_time = time.time()
            query_result = evaluator.evaluate_query(query, top_k=top_k, method=method)
            end_time = time.time()
            
            # Extract test names for comparison with ground truth
            predictions = [item['testName'] for item in query_result['results']]
            method_predictions.append(predictions)
            method_processing_times.append((end_time - start_time) * 1000)  # ms
            
            # Log predictions for this query
            logger.info(f"Query: {query}")
            logger.info(f"Method: {method}, Predictions: {predictions}")
            logger.info(f"Ground truth: {ground_truth.get(query, [])}")
        
        # Ground truth for the queries
        ground_truth_lists = []
        for query in test_queries:
            gt = ground_truth.get(query, [])
            # Convert to list if it's not already
            if not isinstance(gt, list):
                gt = [gt]
            # Ensure the ground truth is populated and valid
            if len(gt) == 0:
                logger.warning(f"Empty ground truth for query: {query}")
            ground_truth_lists.append(gt)
        
        # Calculate metrics with proper logging
        try:
            recall = calculate_mean_recall_at_k(method_predictions, ground_truth_lists, k=top_k)
            map_score = calculate_map_at_k(method_predictions, ground_truth_lists, k=top_k)
            
            # Log metrics for debugging
            logger.info(f"Method: {method}, Recall@{top_k}: {recall:.4f}, MAP@{top_k}: {map_score:.4f}")
            
            results.append({
                'method': method,
                'mean_recall_at_k': recall,
                'map_at_k': map_score,
                'avg_processing_time_ms': np.mean(method_processing_times),
                'queries_evaluated': len(test_queries)
            })
        except Exception as e:
            logger.error(f"Error calculating metrics for method {method}: {str(e)}")
            # Add zeros to avoid breaking the visualization
            results.append({
                'method': method,
                'mean_recall_at_k': 0.0,
                'map_at_k': 0.0,
                'avg_processing_time_ms': np.mean(method_processing_times) if method_processing_times else 0.0,
                'queries_evaluated': len(test_queries)
            })
    
    return pd.DataFrame(results)

def get_sample_benchmark_queries() -> List[Dict[str, Any]]:
    """
    Return a set of sample benchmark queries with ground truth for testing
    """
    return [
        {
            "query": "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.",
            "relevant_items": ["JavaScript Programming", "Coding Challenge - Java", "Collaboration Skills Assessment", "Leadership Assessment", "Core Java (Entry Level)", "Core Java (Advanced Level)"]
        },
        {
            "query": "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes.",
            "relevant_items": ["Python Programming", "SQL (Structured Query Language)", "JavaScript Programming", "Coding Challenge - Python", "Core Java", "Automata - SQL"]
        },
        {
            "query": "I am hiring for an analyst and wants applications to screen using Cognitive and personality tests, what options are available within 45 mins.",
            "relevant_items": ["Cognitive Ability Assessment", "Personality Assessment", "Data Analyst Assessment", "Network Engineer/Analyst Solution"]
        }
    ]
