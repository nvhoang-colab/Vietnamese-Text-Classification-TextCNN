import os
from pyvi import ViTokenizer
import pandas as pd
import pickle
from tqdm import tqdm
from define import *
import numpy as np

class NLP(object):
    def __init__(self, text = None):
        self.text = text
        self.__set_stopwords()

    def __set_stopwords(self):
        with open(STOP_WORDS, 'r') as f:
            self.stopwords = set([w.strip() for w in f.readlines()])

    def segmentation(self):
        return ViTokenizer.tokenize(self.text)

    def split_words(self):
        text = self.segmentation()
        try:
            return [x.strip(SPECIAL_CHARACTER).lower() for x in text.split()]
        except TypeError:
            return []

    def get_text_feature(self):
        split_words = self.split_words()
        words = [word for word in split_words if word.encode('utf-8') not in self.stopwords and word != '']
        text = ' '.join(words)
        return text

def read_data(path):
    print("Read Data")
    data = pd.read_csv(path)
    data['content'] = data['title'] + ' sep ' + data['description']
    data.drop(columns=['title','description'])
    data['content'] = [NLP(data).get_text_feature() for data in data['content']]
    return data

if __name__ == '__main__':
    # read data (csv to dict)
    train = read_data(DATA_TRAIN_CSV)
    val = read_data(DATA_VAL_CSV)
    test = read_data(DATA_TEST_CSV)

    # save data
    train.to_csv(PROCESSED_DATA_TRAIN_CSV,index=False)
    val.to_csv(PROCESSED_DATA_VAL_CSV,index=False)
    test.to_csv(PROCESSED_DATA_TEST_CSV,index=False)




