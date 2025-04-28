import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def load_data(phishing_path, legitimate_path):
    """Load raw datasets"""
    phishing_data = pd.read_csv(phishing_path)
    phishing_data['label'] = 1  # 1 for phishing
    
    legitimate_data = pd.read_csv(legitimate_path)
    legitimate_data['label'] = 0  # 0 for legitimate
    
    # Combine datasets
    combined_data = pd.concat([phishing_data, legitimate_data], ignore_index=True)
    return combined_data

def clean_text(text):
    """Clean and normalize email text"""
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

def preprocess_data(data, text_column):
    """Preprocess the email data"""
    # Clean the text
    data['cleaned_text'] = data[text_column].apply(clean_text)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], 
        data['label'], 
        test_size=0.2, 
        random_state=42
    )
    
    # Create DataFrames for training and testing
    train_df = pd.DataFrame({
        'text': X_train,
        'label': y_train
    })
    
    test_df = pd.DataFrame({
        'text': X_test,
        'label': y_test
    })
    
    return train_df, test_df

if __name__ == "__main__":
    # Paths to raw data
    phishing_path = "Data_set_directory/Raw/Phishing_data.csv"
    legitimate_path = "Data_set_directory/Raw/Legitimate_data.csv"
    
    # Load data
    combined_data = load_data(phishing_path, legitimate_path)
    
    # Preprocess data
    train_df, test_df = preprocess_data(combined_data, 'email_text')  # Replace 'email_text' with actual column name
    
    # Save processed data
    train_df.to_csv("Data_set_directory/Processed/Training_data.csv", index=False)
    test_df.to_csv("Data_set_directory/Processed/Testing_data.csv", index=False)
    
    print("Data preprocessing completed!")
