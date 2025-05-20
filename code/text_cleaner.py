def clean_text(text: str) -> str:
    """
    Clean text by fixing common formatting issues
    Args:
        text: Input text string
    Returns:
        Cleaned text string
    """
    import re
    
    # Remove 'et al.' and similar variations
    text = re.sub(r'\s+(?:and\s+)?et\.?\s*al\.?', '', text)
    
    # Fix cases where a single letter is split from the rest of the word
    # e.g., 'C orona' -> 'Corona'
    text = re.sub(r'\b([A-Z])\s+([a-z]+)\b', r'\1\2', text)
    
    # Fix cases where a word is split at the end
    # e.g., 'huma n' -> 'human'
    text = re.sub(r'\b([a-z]{3,})\s+([a-z])\b(?!\s+[a-z])', r'\1\2', text)
    
    # Fix cases where both parts are short
    # e.g., 'hu man' -> 'human'
    text = re.sub(r'\b([a-z]{2})\s+([a-z]{2,3})\b(?!\s+[a-z])', r'\1\2', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing spaces
    text = text.strip()
    
    return text
