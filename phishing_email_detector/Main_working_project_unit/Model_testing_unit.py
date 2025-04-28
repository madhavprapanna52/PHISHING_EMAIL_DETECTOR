import pandas as pd
import numpy as np
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_extraction_engine import extract_features

def load_hyperparameters():
    """Load hyperparameters from config file"""
    with open("Config/Hyperparameter_tunning.json", "r") as f:
        hyperparams = json.load(f)
    return hyperparams

def train_random_forest(X_train, y_train, hyperparams):
    """Train a Random Forest classifier"""
    rf_params = hyperparams.get("random_forest", {})
    rf_classifier = RandomForestClassifier(
        n_estimators=rf_params.get("n_estimators", 100),
        max_depth=rf_params.get("max_depth", None),
        min_samples_split=rf_params.get("min_samples_split", 2),
        random_state=42
    )
    
    rf_classifier.fit(X_train, y_train)
    return rf_classifier

def train_svm(X_train, y_train, hyperparams):
    """Train an SVM classifier"""
    svm_params = hyperparams.get("svm", {})
    svm_classifier = SVC(
        C=svm_params.get("C", 1.0),
        kernel=svm_params.get("kernel", "rbf"),
        gamma=svm_params.get("gamma", "scale"),
        probability=True,
        random_state=42
    )
    
    svm_classifier.fit(X_train, y_train)
    return svm_classifier

def evaluate_model(model, X_val, y_val):
    """Evaluate model performance"""
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

def save_model(model, vectorizer):
    """Save the trained model and vectorizer"""
    model_data = {
        "model": model,
        "vectorizer": vectorizer
    }
    
    joblib.dump(model_data, "Models_data/Trained_model.pkl")
    
    # Save model architecture info
    model_info = {
        "model_type": type(model).__name__,
        "feature_count": model.n_features_in_ if hasattr(model, "n_features_in_") else "Unknown",
        "feature_importance": model.feature_importances_.tolist() if hasattr(model, "feature_importances_") else None
    }
    
    with open("Models_data/Model_artitechture.json", "w") as f:
        json.dump(model_info, f, indent=4)

if __name__ == "__main__":
    # Load training data
    train_df = pd.read_csv("Data_set_directory/Processed/Training_data.csv")
    
    # Extract features
    vectorizer, features = extract_features(train_df['text'], mode='train')
    labels = train_df['label']
    
    # Load hyperparameters
    hyperparams = load_hyperparameters()
    
    # Train models
    rf_model = train_random_forest(features, labels, hyperparams)
    svm_model = train_svm(features, labels, hyperparams)
    
    # Evaluate models on validation set (assuming a split within training data)
    # In a real scenario, you would split the training data into train and validation sets
    rf_metrics = evaluate_model(rf_model, features, labels)
    svm_metrics = evaluate_model(svm_model, features, labels)
    
    print("Random Forest metrics:", rf_metrics)
    print("SVM metrics:", svm_metrics)
    
    # Select the best model (based on F1 score)
    best_model = rf_model if rf_metrics["f1_score"] > svm_metrics["f1_score"] else svm_model
    
    # Save the best model
    save_model(best_model, vectorizer)
    
    print(f"Best model ({type(best_model).__name__}) saved successfully!")
