import joblib
import pandas as pd
import argparse
from feature_extraction_engine import extract_features

def load_model():
    """Load the trained model and vectorizer""" # using trainned model ? 
    try:
        model_data = joblib.load("Models_data/Trained_model.pkl")
        model = model_data["model"]
        vectorizer = model_data["vectorizer"]
        return model, vectorizer
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def classify_email(email_text, model, vectorizer):
    """Classify the email as phishing or legitimate"""
    if not email_text.strip():
        return "ERROR: Empty email content"
    
    # Convert email to DataFrame for feature extraction
    email_df = pd.DataFrame({'text': [email_text]})
    
    # Extract features
    try:
        _, features = extract_features(email_df['text'], mode='test', vectorizer=vectorizer)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        if prediction == 1:
            confidence = probability[1] * 100
            result = f"PHISHING (Confidence: {confidence:.2f}%)"
        else:
            confidence = probability[0] * 100
            result = f"LEGIT (Confidence: {confidence:.2f}%)"
        
        return result
    except Exception as e:
        return f"ERROR: Failed to classify email - {e}"

def main():
    parser = argparse.ArgumentParser(description='Phishing Email Detector')
    parser.add_argument('--file', type=str, help='Path to file containing email text')
    args = parser.parse_args()
    
    # Load the model
    model, vectorizer = load_model()
    if model is None or vectorizer is None:
        print("Failed to load the model. Please make sure the model is trained.")
        return
    
    if args.file:
        # Read email from file
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                email_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
    else:
        # Interactive mode
        print("Paste the email content below and press Ctrl+D (Unix) or Ctrl+Z (Windows) when finished:")
        email_lines = []  #TODO make an good while loop structure for command line interaction unit 
        try:
            while True:
                line = input()
                email_lines.append(line)
        except EOFError:
            email_text = "\n".join(email_lines)
    
    # Classify the email
    result = classify_email(email_text, model, vectorizer)
    print("\nClassification Result:")
    print(result)

if __name__ == "__main__":
    main()
