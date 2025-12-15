import re

def highlight_keywords(text):
    # Example spam keywords
    spam_keywords = ["win", "free", "offer", "money", "urgent", "click", "limited", "prize"]
    
    matched_keywords = []
    highlighted_text = text

    for word in spam_keywords:
        pattern = re.compile(rf"\b{word}\b", re.IGNORECASE)
        if pattern.search(text):
            matched_keywords.append(word)
            highlighted_text = pattern.sub(f"**:red[{word}]**", highlighted_text)

    # ✅ Always return 2 values
    return matched_keywords, highlighted_text
