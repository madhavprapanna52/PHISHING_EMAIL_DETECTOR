# scripts/train_model.py

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def has_mean(t):
    words = str(t).split()
    return any(word.lower() not in ENGLISH_STOP_WORDS for word in words)


def train_and_save_model(data_path: str, model_path: str):
    df = pd.read_csv(data_path)
    # final droping commands for train set 
    df.dropna(subset=['Email Text', 'Email Type'], inplace=True)
    # Making emails labels 
    df['Email Type'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
    vectorizer = CountVectorizer(stop_words='english', max_features=3000)
    X = vectorizer.fit_transform(df['Email Text'])
    y = df['Email Type']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    print("Evaluation:\n", classification_report(y_test, model.predict(X_test)))

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, model_path.replace("model", "vectorizer"))

if __name__ == "__main__":
    train_and_save_model("/home/madhavr/Desktop/Yantragya_project_1.0/Data_set.csv", "/home/madhavr/Desktop/Yantragya_project_1.0/Scam_detection_system/Project_files/logistic_model.pkl")
