'''
Data preprocessing unit 
    1. Cleaning dataset 
    2. Processing with obvious intelligence text engines for reducing data load
    3. Stemming, lemmatization, tokenizing words

Process layout 
Dataset reading loop for efficiency and crash prevention 
Cleaning dataset throughout 
'''
import pandas as pd
from My_basic_tool import *  # assuming your own utilities
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import csv

# Load dataset with safe fallback for NaNs
file_path = '/home/madhavr/Desktop/Yantragya_project_1.0/Data_set.csv'
try:
    data_set = pd.read_csv(file_path)
    print("Original dataset size:", data_set.shape)
except Exception as e:
    print("Failed to read dataset:", e)
    exit()

# Drop completely empty rows only
data_set.dropna(subset=['Email Text', 'Email Type'], inplace=True)

# See sample
print('Checking cleaned dataset:')
print(data_set.head(5))

# Define text cleaning function
def clean_text(text, use_stemming=False):
    """
    Clean dataset: remove punctuation, HTML, links, unwanted chars.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""
    
    try:
        text = re.sub(r'http\S+|www\S+|@\S+', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower()
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]

        if use_stemming:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]

        return ' '.join(tokens)
    
    except Exception as e:
        print("Text cleaning error:", e)
        return ""

# Output file path
output_file = '/home/madhavr/Desktop/Yantragya_project_1.0/Scam_detection_system/Data_set/Cleanned_dataset.csv'

# Write cleaned dataset
with open(output_file, 'w', newline='', encoding='utf-8') as cleaned_dataset:
    writer = csv.writer(cleaned_dataset)
    writer.writerow(['Email Text', 'Email Type'])

    for idx, row in data_set.iterrows():
        text = row.get('Email Text', '')
        label = row.get('Email Type', '')

        processed_text = clean_text(text, use_stemming=True)

        print('_'*50)
        print(f'Writing cleaned text: {processed_text[:50]}...')

        writer.writerow([processed_text, label])
