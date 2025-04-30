'''
Data preprocesssing unit 
    1. Cleanning dataset 
    2. Processing with obvious intellegence text engines for reducing data load
    3. Stemming , lemmatisation , tokenising words

Process layout 
Dataset reading loop for efficiency and crash prevention 
cleanning dataset through out 
'''
import pandas as pd
from My_basic_tool import *
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
import matplotlib.pyplot as plt
import csv 

# Loading data 
data_set = pd.read_csv('/home/madhavr/Desktop/Yantragya_project_1.0/Data_set.csv')  # working till 1000
data_set = data_set.dropna()

# cleanning data set through manual pic 


# cleanning dataset 
def clean_text(text, use_stemming=False):
    """
    clean dataset 
    removing elements = [punctuations , html, rare words, too frquent_word, irrelevent specials]
    """
    # processing emails --
    


    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text(separator='\n') # concatination with new line 
    text = re.sub(r'[^a-zA-Z0-9\s.,?!-]', '', text).strip()
    text = re.sub(r'http\S+|www\S+|@\S+', '', text)
    text = text.strip()
    if isinstance(text, str):
        text = text.lower()
        stop_words = set(stopwords.words('english'))  #  stopwords
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if not((word in string.punctuation) and (word in stop_words))]  # if word isnt puct/stop - gets further

        if use_stemming:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)  # making spaced output tokens 
    return ''

# Cleanned dataset loading


with open('/home/madhavr/Desktop/Yantragya_project_1.0/Scam_detection_system/Data_set/Cleanned_dataset.csv', 'w') as cleaned_dataset:
    writer = csv.writer(cleaned_dataset)
    head = ('Email Text','Email Type')
    writer.writerow(head)

    for index, row in data_set.iterrows():
        text = row['Email Text']
        label = row["Label"]
        processed_text = clean_text(text)
        e = [processed_text, label]
        writer.writerow(e)

d = pd.read_csv('/home/madhavr/Desktop/Yantragya_project_1.0/Scam_detection_system/Data_set/Cleanned_dataset.csv')
print(d.head(20))

