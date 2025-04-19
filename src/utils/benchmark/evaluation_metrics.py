
"""
Evaluation metrics for assessing recommendation quality
"""
import numpy as np
import logging
from typing import List
from src.utils.benchmark.text_match import clean_text_for_comparison, is_substantial_match

logger = logging.getLogger(__name__)

def calculate_mean_recall_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: int = 3) -> float:
    """Calculate Mean Recall@K metric"""
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
                
                if is_substantial_match(pred_clean, truth_clean):
                    relevant_retrieved += 1
                    matches_found.append((pred_item, truth_item))
                    break
        
        recall = relevant_retrieved / len(truth)
        recall_values.append(recall)
        
        if matches_found:
            logger.info(f"Query index {i}: Found {relevant_retrieved} relevant items out of {len(truth)} ground truth items")
            for match in matches_found:
                logger.info(f"  - '{match[0]}' matched with '{match[1]}'")
        else:
            logger.warning(f"Query index {i}: No matches found!")
    
    mean_recall = np.mean(recall_values) if recall_values else 0.0
    logger.info(f"Mean Recall@{k}: {mean_recall:.4f}")
    return mean_recall

def calculate_map_at_k(predictions: List[List[str]], ground_truth: List[List[str]], k: int = 3) -> float:
    """Calculate Mean Average Precision@K (MAP@K) metric"""
    if len(predictions) != len(ground_truth):
        raise ValueError("Predictions and ground truth must have the same length")
    
    ap_values = []
    
    for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
        if not truth:
            continue
            
        pred_k = pred[:k]
        precision_values = []
        num_relevant_found = 0
        
        for j, pred_item in enumerate(pred_k):
            pred_clean = clean_text_for_comparison(pred_item)
            is_relevant = False
            
            for truth_item in truth:
                truth_clean = clean_text_for_comparison(truth_item)
                
                if is_substantial_match(pred_clean, truth_clean):
                    is_relevant = True
                    logger.info(f"MAP - Match found at position {j+1}: '{pred_item}' matches '{truth_item}'")
                    break
            
            if is_relevant:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (j + 1)
                precision_values.append(precision_at_i)
        
        ap = sum(precision_values) / len(truth) if precision_values else 0
        ap_values.append(ap)
        
        logger.info(f"Query index {i}: AP = {ap:.4f}, found {num_relevant_found} relevant items")
    
    mean_ap = np.mean(ap_values) if ap_values else 0.0
    logger.info(f"Mean AP@{k}: {mean_ap:.4f}")
    return mean_ap

