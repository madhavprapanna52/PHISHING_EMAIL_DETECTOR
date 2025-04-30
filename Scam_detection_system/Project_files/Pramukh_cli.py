import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer

# === Define paths ===
model_path = "/home/madhavr/Desktop/Yantragya_project_1.0/Scam_detection_system/Project_files/logistic_model.pkl"
vectorizer_path = model_path.replace("model", "vectorizer")

# === Load model and vectorizer ===
if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
    print("Error: Model or vectorizer file not found. Check paths.")
    exit()

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

def predict_email(text: str):
    if not text.strip():
        print("Non empty mails are supposed to be processed here")
        return
    
    # Vectorize the text using loaded vectorizer
    text_vector = vectorizer.transform([text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    confidence = model.predict_proba(text_vector)[0][prediction]

    # Show result
    label = "Phishing Email 🚨" if prediction == 1 else "Safe Email ✅"
    print(f"\n📩 Prediction: {label}")
    print(f"🔍 Confidence: {confidence*100:.2f}%\n")

if __name__ == "__main__":
    print("🛡️ Scam Email Classifier CLI")
    print("Type an email to test it (Ctrl+C to quit)\n")

    while True:
        try:
            user_input = input("✉️  Email Text: ")
            predict_email(user_input)
        except KeyboardInterrupt:
            print("\n👋 Exiting CLI. Stay safe and alert!")
            break
