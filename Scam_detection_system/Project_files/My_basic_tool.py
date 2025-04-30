import pandas as pd
import csv


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


def Manual_clean(path_to_file):
    file = open('Data_set.csv', 'w')
    w = csv.writer(file)
    header = ('Email Text','Label')
    w.writerow(header)
    with open(path_to_file, 'r') as data:
        reader = csv.reader(data)
        for row in reader:
            # Taking manuals to get out just clean dataset
            words = Extract_words(row[1])
            if len(words) > 10:
                element = (words, row[2])
                w.writerow(element)

