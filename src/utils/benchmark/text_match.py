
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
    text = re.sub(r'\s*test\s*', ' ', text)
    text = re.sub(r'\s*challenge\s*', ' ', text)
    
    text = re.sub(r'core\s+(\w+)', r'\1', text)  # "Core Java" -> "Java"
    
    text = text.replace('javascript', 'java script').replace('js', 'java script')
    text = text.replace('collab', 'collaborat') 
    text = text.replace('cognitive ability', 'cognitive')
    text = text.replace('coding', 'programming')
    text = text.replace('data analyst', 'data analysis')
    text = text.replace('analyst', 'analysis')
    
    text = text.replace('python', 'python programming')
    text = text.replace('java ', 'java programming ')
    text = text.replace('sql', 'database sql')
    text = text.replace('lead', 'leadership')
    text = text.replace('manage', 'management')
    text = text.replace('eng', 'engineer engineering')
    
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def is_substantial_match(text1: str, text2: str) -> bool:
    """
    Determine if two cleaned strings are substantially matching with increased leniency
    """
    logger.info(f"Comparing: '{text1}' with '{text2}'")
    
    if text1 in text2 or text2 in text1:
        logger.info(f"✓ Substring match found between '{text1}' and '{text2}'")
        return True
    
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    key_terms = [
        # Programming languages
        'java', 'python', 'sql', 'javascript', 'js', 'c++', 'csharp', 'ruby', 'php',
        
        # Skills and assessment types
        'cognitive', 'personality', 'collaboration', 'leadership', 'data', 'analyst',
        'analysis', 'coding', 'programming', 'challenge', 'interview', 'profiling',
        'develop', 'engineer', 'technical', 'problem', 'solving',
        
        # Business domains
        'business', 'management', 'finance', 'marketing', 'sales', 'service',
        
        # Soft skills
        'communication', 'teamwork', 'critical', 'thinking', 'creativity', 'verbal',
        'numerical', 'reasoning', 'logical', 'emotional', 'intelligence',
        
        # Role keywords
        'developer', 'designer', 'manager', 'executive', 'assistant', 'representative'
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
        if overlap_ratio >= 0.15:
            logger.info(f"✓ Word overlap ratio {overlap_ratio:.2f} with common words: {common_words}")
            return True
    
    for word1 in words1:
        if len(word1) <= 3:
            continue 
        for word2 in words2:
            if len(word2) <= 3:
                continue

            if (word1 in word2 or word2 in word1 or 
                (len(word1) > 3 and len(word2) > 3 and  
                 (word1[:3] == word2[:3] or 
                  word1[-3:] == word2[-3:]))): 
                logger.info(f"✓ Similar words found: '{word1}' ~ '{word2}'")
                return True
    
    # No substantial match found
    logger.info(f"✗ No match between '{text1}' and '{text2}'")
    return False
