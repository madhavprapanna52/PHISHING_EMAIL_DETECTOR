import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction_engine import extract_features

def load_model():
    """Load the trained model and vectorizer"""
    model_data = joblib.load("Models_data/Trained_model.pkl")
    model = model_data["model"]
    vectorizer = model_data["vectorizer"]
    return model, vectorizer

def test_model(model, vectorizer, test_df):
    """Test the model on test data"""
    # Extract features using the trained vectorizer
    _, features = extract_features(test_df['text'], mode='test', vectorizer=vectorizer)
    
    # Make predictions
    y_pred = model.predict(features)
    y_true = test_df['label']
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=['Legitimate', 'Phishing'])
    
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    return report, cm, y_pred

def plot_confusion_matrix(cm):
    """Plot confusion matrix as a heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('NOTEBOOKS/confusion_matrix.png')
    plt.close()

def save_results(test_df, y_pred, report):
    """Save test results"""
    # Add predictions to test data
    test_df['predicted'] = y_pred
    
    # Save test results
    test_df.to_csv("Data_set_directory/Processed/test_results.csv", index=False)
    
    # Save classification report
    with open("NOTEBOOKS/classification_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    # Load test data
    test_df = pd.read_csv("Data_set_directory/Processed/Testing_data.csv")
    
    # Load trained model and vectorizer
    model, vectorizer = load_model()
    
    # Test the model
    report, cm, y_pred = test_model(model, vectorizer, test_df)
    
    # Print classification report
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)
    
    # Save results
    save_results(test_df, y_pred, report)
    
    print("Model testing completed!")
