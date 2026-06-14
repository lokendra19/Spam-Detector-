import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class TextFeatures(BaseEstimator, TransformerMixin):
    """Hand-crafted spam signals that TF-IDF alone misses."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        feats = []
        for text in X:
            words = text.split()
            total = max(len(words), 1)
            caps_ratio   = sum(1 for w in words if w.isupper() and len(w) > 1) / total
            url_count    = len(re.findall(r'https?://\S+|www\.\S+', text))
            dollar_signs = len(re.findall(r'[\$£€₹]', text))
            excl_marks   = text.count('!')
            digits_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
            phone_nums   = len(re.findall(r'\b\d{10,}\b|\b0\d{9}\b|\b1-\d{3}', text))
            urgent_words = len(re.findall(
                r'\b(urgent|act now|limited|expires|immediately|verify|suspended|'
                r'blocked|claim|winner|congratulations|free|prize)\b', text, re.I))
            feats.append([caps_ratio, url_count, dollar_signs, excl_marks,
                          digits_ratio, phone_nums, urgent_words])
        return np.array(feats, dtype=float)
