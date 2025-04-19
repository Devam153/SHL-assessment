
"""
Text matching utilities for comparing assessment names and descriptions
"""
import re
import logging

logger = logging.getLogger(__name__)

def clean_text_for_comparison(text: str) -> str:
    """
    Clean and normalize text for more accurate comparison
    """
    if not isinstance(text, str):
        text = str(text)
    
    text = text.lower()
    
    text = re.sub(r'\s*\(new\)\s*', ' ', text)
    text = re.sub(r'\s*assessment\s*', ' ', text)
    text = re.sub(r'\s*solution\s*', ' ', text)
    
    text = re.sub(r'core\s+(\w+)', r'\1', text)  # "Core Java" -> "Java"
    
    text = text.replace('javascript', 'java script').replace('js', 'java script')
    text = text.replace('collab', 'collaborat')  # Match collaboration/collaborate/collaborative
    text = text.replace('cognitive ability', 'cognitive')
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_substantial_match(text1: str, text2: str) -> bool:
    """
    Determine if two cleaned strings are substantially matching
    """
    logger.info(f"Comparing: '{text1}' with '{text2}'")
    
    if text1 in text2 or text2 in text1:
        logger.info(f"✓ Substring match found between '{text1}' and '{text2}'")
        return True
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    key_terms = [
        'java', 'python', 'sql', 'javascript', 'cognitive', 'personality', 
        'collaboration', 'leadership', 'data', 'analyst', 'challenge',
        'coding', 'assessment', 'profiling', 'interview'
    ]
    
    common_key_terms = [term for term in key_terms if term in text1 and term in text2]
    if common_key_terms:
        logger.info(f"✓ Common key terms found: {common_key_terms}")
        return True
    
    common_words = words1.intersection(words2)
    
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
    
    logger.info(f"✗ No match between '{text1}' and '{text2}'")
    return False
