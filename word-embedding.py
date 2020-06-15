import os
from gensim import corpora, matutils
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from tqdm import tqdm
from define import *
import numpy as np


class FeatureExtraction(object):
    def __init__(self, data, dataNLP = False):
        self.data = data      

    def __build_dictionary(self):
        print('Building dictionary')
        dict_words = []
        for text in tqdm(self.data):
            words = text['content'].split(' ')
            dict_words.append(words)
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=100, no_above=0.5)
        dictionary.save_as_text(DICTIONARY_PATH)

    def __build_modelw2v(self):
        print('Buiding Word2Vec')
        sentences = []
        for text in tqdm(self.data):
            words = text['content'].split(' ')
            sentences.append(words)
        model = Word2Vec(sentences=sentences,
                          workers = 4,
                          min_count = 2,
                          size = self.max_features,
                          window = 10)
        model.save(WORD2VEC_PATH)

    def __load_dictionary(self):
        if os.path.exists(DICTIONARY_PATH) == False:
            self.__build_dictionary()
        self.dictionary = corpora.Dictionary.load_from_text(DICTIONARY_PATH)

    def __load_modelw2v(self):
        if os.path.exists(WORD2VEC_PATH) == False:
            self.__build_modelw2v()
        self.word2vec = Word2Vec.load(WORD2VEC_PATH)

    def __build_bow_dataset(self):
        print('Building dataset')
        self.features = []
        self.labels = []
        for d in tqdm(self.data):
            self.features.append(self.get_dense_bow(d['content']))
            self.labels.append(d['category'])

    def __build_w2v_dataset(self):
        print('Building dataset')        
        self.features = []
        self.labels = []
        self.__load_modelw2v()
        for d in tqdm(self.data):
            words = d['content'].split(' ')
            vec = []
            for i in range(self.max_len):
                try:
                    vec.append(self.word2vec.wv[words[i]])
                except IndexError:
                    vec.append(np.zeros([self.max_features]))
                except KeyError:
                    vec.append(np.zeros([self.max_features])) 
            self.features.append(vec)
            self.labels.append(d['category'])
            
    def get_dense_bow(self, text):
        words = text.split()
        self.__load_dictionary()
        vec = self.dictionary.doc2bow(words)
        dense = list(matutils.corpus2dense([vec], num_terms=len(self.dictionary)).T[0])
        return dense
        
    def get_data_and_label_tfidf(self):
        print('Building dataset')
        self.features = []
        self.labels = []
        for d in tqdm(self.data):
            self.features.append(d['content'])
            self.labels.append(d['category'])        
        return self.features , self.labels

    def get_data_and_label_w2v(self, max_len = 60, max_features = 300):
        self.max_len = max_len
        self.max_features = max_features
        self.__build_w2v_dataset()
        return self.features, self.labels
    
    def get_data_and_label_bow(self):
        self.__build_bow_dataset()
        return self.features, self.labels

    def read_feature(self):
        return self.data['features'], self.data['labels']


def get_feature_dict(value_features,value_labels):
    return {
            "features":value_features,
            "labels":value_labels
        }

if __name__ == '__main__':
    # read data (csv to dict)
    train = pd.read_csv(PROCESSED_DATA_TRAIN_CSV).to_dict('record')
    val = pd.read_csv(PROCESSED_DATA_VAL_CSV).to_dict('record')
    test = pd.read_csv(PROCESSED_DATA_TEST_CSV).to_dict('record')

    # # tfidf features
    # features_train, labels_train = FeatureExtraction(data=train).get_data_and_label_tfidf()
    # features_val, labels_val = FeatureExtraction(data=val).get_data_and_label_tfidf()
    # features_test, labels_test = FeatureExtraction(data=test).get_data_and_label_tfidf()

    # vectorizer = TfidfVectorizer(use_idf=True,
    #                              smooth_idf=True,
    #                              sublinear_tf=False,
    #                              max_features = 12000,
    #                              min_df=0.0, max_df=1.0,
    #                              ngram_range=(1, 2))
    # features_train = vectorizer.fit_transform(features_train).todense()
    # features_val = vectorizer.transform(features_val).todense()
    # features_test = vectorizer.transform(features_test).todense()

    # # save
    # features_train_dict = get_feature_dict(features_train, labels_train)
    # features_test_dict = get_feature_dict(features_test, labels_test)
    # features_val_dict = get_feature_dict(features_val, labels_val)

    # pickle.dump(features_train_dict, open(FEATURES_TRAIN_TFIDF.replace(".p","12K.p"), "wb"))
    # pickle.dump(features_test_dict, open(FEATURES_TEST_TFIDF.replace(".p","12K.p"), "wb"))
    # pickle.dump(features_val_dict, open(FEATURES_VAL_TFIDF.replace(".p","12K.p"), "wb"))
    # pickle.dump(vectorizer, open('features/vec_tfidf.p',"wb"))

    # # bow features
    # features_train, labels_train = FeatureExtraction(data=train).get_data_and_label_bow()
    # features_test, labels_test = FeatureExtraction(data=test).get_data_and_label_bow()
    # features_val, labels_val = FeatureExtraction(data=val).get_data_and_label_bow()

    # # save
    # features_train_dict = get_feature_dict(features_train, labels_train)
    # features_test_dict = get_feature_dict(features_test, labels_test)
    # features_val_dict = get_feature_dict(features_val, labels_val)

    # pickle.dump(features_train_dict, open(FEATURES_TRAIN_BOW, "wb"))
    # pickle.dump(features_test_dict, open(FEATURES_TEST_BOW, "wb"))
    # pickle.dump(features_val_dict, open(FEATURES_VAL_BOW, "wb"))

    # word2vec features
    features_train, labels_train = FeatureExtraction(data=train).get_data_and_label_w2v()
    features_test, labels_test = FeatureExtraction(data=test).get_data_and_label_w2v()
    features_val, labels_val = FeatureExtraction(data=val).get_data_and_label_w2v()

    # save
    features_train_dict = get_feature_dict(features_train, labels_train)
    features_test_dict = get_feature_dict(features_test, labels_test)
    features_val_dict = get_feature_dict(features_val, labels_val)

    pickle.dump(features_train_dict, open(FEATURES_TRAIN_W2V.replace(".p",".p"), "wb"))
    pickle.dump(features_test_dict, open(FEATURES_TEST_W2V.replace(".p",".p"), "wb"))
    pickle.dump(features_val_dict, open(FEATURES_VAL_W2V.replace(".p",".p"), "wb"))
    pass
