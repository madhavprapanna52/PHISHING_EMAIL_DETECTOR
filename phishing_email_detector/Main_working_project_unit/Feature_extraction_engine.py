import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

def extract_text_features(text_series):
    """Extract TF-IDF features from email text"""
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,  # Limit features to prevent overfitting
        stop_words='english',
        ngram_range=(1, 2)  # Include unigrams and bigrams
    )
    
    # Fit and transform the text data
    tfidf_features = tfidf_vectorizer.fit_transform(text_series)
    
    return tfidf_vectorizer, tfidf_features

def extract_url_features(text_series):
    """Extract URL-related features from emails"""
    # Count URLs in each email
    url_counts = text_series.apply(lambda x: len(re.findall(r'https?://\S+|www\.\S+', x)))
    
    # Check for suspicious TLDs
    suspicious_tlds = ['.info', '.xyz', '.top', '.tk', '.ml']
    suspicious_tld_counts = text_series.apply(
        lambda x: sum(1 for tld in suspicious_tlds if tld in x.lower())
    )
    
    # Return as DataFrame
    url_features = pd.DataFrame({
        'url_count': url_counts,
        'suspicious_tld_count': suspicious_tld_counts
    })
    
    return url_features

def extract_linguistic_features(text_series):
    """Extract linguistic features from emails"""
    # Count number of tokens
    token_counts = text_series.apply(lambda x: len(word_tokenize(x)) if isinstance(x, str) else 0)
    
    # Count special characters
    special_char_counts = text_series.apply(
        lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', x)) if isinstance(x, str) else 0
    )
    
    # Check for urgent words
    urgent_words = ['urgent', 'immediately', 'alert', 'attention', 'important']
    urgent_word_counts = text_series.apply(
        lambda x: sum(1 for word in urgent_words if word in x.lower().split()) if isinstance(x, str) else 0
    )
    
    # Return as DataFrame
    linguistic_features = pd.DataFrame({
        'token_count': token_counts,
        'special_char_count': special_char_counts,
        'urgent_word_count': urgent_word_counts
    })
    
    return linguistic_features

def combine_features(tfidf_features, url_features, linguistic_features):
    """Combine all features into a single feature matrix"""
    # Convert sparse matrix to dense array
    tfidf_array = tfidf_features.toarray()
    
    # Combine with other features
    url_array = url_features.values
    linguistic_array = linguistic_features.values
    
    # Concatenate all features
    combined_features = np.hstack((tfidf_array, url_array, linguistic_array))
    
    return combined_features

def extract_features(text_series, mode='train', vectorizer=None):
    """Extract all features from email text"""
    if mode == 'train':
        # For training data, fit and transform
        tfidf_vectorizer, tfidf_features = extract_text_features(text_series)
    else:
        # For test/new data, use existing vectorizer
        tfidf_features = vectorizer.transform(text_series)
        tfidf_vectorizer = vectorizer
    
    # Extract other features
    url_features = extract_url_features(text_series)
    linguistic_features = extract_linguistic_features(text_series)
    
    # Combine all features
    combined_features = combine_features(tfidf_features, url_features, linguistic_features)
    
    return tfidf_vectorizer, combined_features

if __name__ == "__main__":
    # Load training data
    train_df = pd.read_csv("Data_set_directory/Processed/Training_data.csv")
    
    # Extract features
    vectorizer, features = extract_features(train_df['text'], mode='train')
    
    print(f"Extracted {features.shape[1]} features from {features.shape[0]} emails")
