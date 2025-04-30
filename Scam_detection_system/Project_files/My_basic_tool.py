import pandas as pd
import csv
import sys


def encode(data_set_path):
    with pd.read_csv('data_set_path') as df:
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df['Email Type'] = df['Email Type'].map({'Safe Email': 0, 'Phishing': 1})  # 0 = Safe, 1 = Phishing


# Basic preprocessing 
def Extract_words(text):
    words_list = []
    l = text.split(' ')
    # filterring just the words 
    for word in l:
        its_word = 1
        for _ in word:
            # Making sure the whole word is word 
            if _.isalpha():
                its_word = 1
            else:
                its_word *= 0
        if its_word:
            words_list.append(word)
    # Process these words list with english subwords 
    return ' '.join(words_list)
csv.field_size_limit(sys.maxsize)

def Manual_clean(path_to_file):
    file = open('Data_set.csv', 'w')
    w = csv.writer(file)
    
    i = 0
    with open(path_to_file, 'r') as data:
        reader = csv.reader(data)
        processing = True
        print(type(reader))
        for i in range(1000000):
            try:
                row = next(reader)  # iterating row 
                text = Extract_words(row[1])
                print(text)
                # Elements addition with filter 
                words_check = not(text == None)
                label_check = not(row[2] == None)
                if words_check and label_check:
                    elem = (text, row[2])
                    w.writerow(elem)
            except StopIteration:
                break
    
def breaker(i=5):
    for _ in range(i):
        if _ == i:
            return 'break'

Manual_clean('/home/madhavr/Desktop/Yantragya_project_1.0/Scam_detection_system/Data_set/Phishing_Email.csv')