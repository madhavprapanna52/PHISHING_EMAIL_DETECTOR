from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)

    def fit_transform(self, texts):
        return self.vectorizer.fit_transform(texts)

    def transform(self, texts):
        return self.vectorizer.transform(texts)

    def get_vectorizer(self):
        return self.vectorizer
