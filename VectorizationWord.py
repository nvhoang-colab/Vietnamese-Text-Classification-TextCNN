import os
from gensim import corpora, matutils
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import pickle
from tqdm import tqdm
from define import *
import numpy as np


class FeatureExtraction(object):
    def __init__(self, data, train = False):
        self.data = data
        self.train = train      

    def __build_modelw2v(self, max_features = 300, weighed = False):
        print('Building Word2Vec')
        sentences = []
        for text in tqdm(self.data):
            words = text.split(" ")
            sentences.append(words)
        model = Word2Vec(sentences=sentences,
                          workers = 4,
                          min_count = 0,
                          size = max_features,
                          sg = 1,
                          window = 10)
        if weighed:
            model.save(WORD2VEC_PATH.replace('.model','weighed.model'))
        else:
            model.save(WORD2VEC_PATH)

    def __build_modeltfidf(self, max_features = 12000, weighed = False):
        print('Building TF-IDF')
        ngram_range = (1, 2)
        if weighed:
            ngram_range = (1, 1)
            max_features = None
        tfidf = TfidfVectorizer(use_idf=True,
                                 smooth_idf=True,
                                 sublinear_tf=False,
                                 max_features = max_features,
                                 min_df=0.0, max_df=1.0,
                                 ngram_range=ngram_range)
        tfidf.fit(self.data)
        if weighed:
            pickle.dump(tfidf, open(TFIDF_PATH.replace('.p','weighed.p'), "wb"))
        else:
            pickle.dump(tfidf, open(TFIDF_PATH, "wb"))

    def __load_modelw2v(self):
        if os.path.exists(WORD2VEC_PATH) == False and self.train:
            self.__build_modelw2v()
        self.word2vec = Word2Vec.load(WORD2VEC_PATH)

    def __load_modeltfidf(self):
        if os.path.exists(TFIDF_PATH) == False and self.train:
            self.__build_modeltfidf()
        self.tfidf = pickle.load(open(TFIDF_PATH, "rb"))

    def __load_modelwwv(self):
        if os.path.exists(TFIDF_PATH.replace('.p','weighed.p')) == False and self.train:
            self.__build_modeltfidf(weighed = True)
        self.word2vec = Word2Vec.load(WORD2VEC_PATH)
        self.tfidf = pickle.load(open(TFIDF_PATH.replace('.p','weighed.p'), "rb"))

    def __build_w2v_features(self, max_len = 40):
        print('Building W2V features')        
        self.features = []
        self.__load_modelw2v()
        for d in tqdm(self.data):
            words = d.split(" ")
            vec = []
            for i in range(max_len):
                try:
                    vec.append(self.word2vec.wv[words[i]])
                except KeyError:
                    vec.append(np.zeros(300))
                except IndexError:
                    vec.append(np.zeros(300))
            self.features.append(vec)

    def __build_tfidf_features(self):
        print('Building TF-IDF features')
        self.features = []
        self.__load_modeltfidf()
        self.features = self.tfidf.transform(self.data).todense()

    def __build_wwv_features(self, max_len = 40):
        print('Building WWV features')
        self.features = []
        self.__load_modelwwv()
        vocal = self.tfidf.vocabulary_
        for d in tqdm(self.data):
            d_tfidf = self.tfidf.transform([d])
            words = d.split()
            vecs = []
            for i in range(max_len):
                try:
                    w2v = np.asarray(self.word2vec.wv[words[i]])
                except KeyError:
                    w2v = np.random.normal(0, 0.05, 300)
                except IndexError:
                    w2v = np.random.normal(0, 0.05, 300)
                try:
                    tfidf = d_tfidf[0, vocal[words[i]]]
                except KeyError:
                    tfidf = np.random.normal(0, 0.05, 1)[0]
                except IndexError:
                    tfidf = np.random.normal(0, 0.05, 1)[0]
                vec = tfidf * w2v
                vecs.append(vec)
            self.features.append(vecs)

    def get_features_w2v(self):
        self.__build_w2v_features()
        return self.features
       
    def get_features_tfidf(self):
        self.__build_tfidf_features()
        return self.features

    def get_features_wwv(self):
        self.__build_wwv_features()
        return self.features


if __name__ == '__main__':
    # # read data
    # train = pd.read_csv(PROCESSED_DATA_TRAIN_CSV)
    # val = pd.read_csv(PROCESSED_DATA_VAL_CSV)
    # test = pd.read_csv(PROCESSED_DATA_TEST_CSV)

    # # tfidf features
    # features_train = FeatureExtraction(data=train['content'], train = True).get_features_tfidf()
    # features_val = FeatureExtraction(data=val['content']).get_features_tfidf()
    # features_test = FeatureExtraction(data=test['content']).get_features_tfidf()

    # # save features
    # pickle.dump(features_train, open(FEATURES_TRAIN_TFIDF.replace(".p",".p"), "wb"))
    # pickle.dump(features_test, open(FEATURES_TEST_TFIDF.replace(".p",".p"), "wb"))
    # pickle.dump(features_val, open(FEATURES_VAL_TFIDF.replace(".p",".p"), "wb"))

    # # w2v features
    # features_train = FeatureExtraction(data=train['content'], train = True).get_features_w2v()
    # features_val = FeatureExtraction(data=val['content']).get_features_w2v()
    # features_test = FeatureExtraction(data=test['content']).get_features_w2v()

    # # save features
    # pickle.dump(features_train, open(FEATURES_TRAIN_W2V, "wb"))
    # pickle.dump(features_test, open(FEATURES_TEST_W2V, "wb"))
    # pickle.dump(features_val, open(FEATURES_VAL_W2V, "wb"))

    # # wwv features
    # features_train = FeatureExtraction(data=train['content'], train = True).get_features_wwv()
    # features_val = FeatureExtraction(data=val['content']).get_features_wwv()
    # features_test = FeatureExtraction(data=test['content']).get_features_wwv()

    # # save features
    # pickle.dump(features_train, open(FEATURES_TRAIN_WWV, "wb"))
    # pickle.dump(features_test, open(FEATURES_TEST_WWV, "wb"))
    # pickle.dump(features_val, open(FEATURES_VAL_WWV, "wb"))

    # save label
    print('Save label')
    pickle.dump(train['category'], open(LABEL_TRAIN, "wb"))
    pickle.dump(test['category'], open(LABEL_TEST, "wb"))
    pickle.dump(val['category'], open(LABEL_VAL, "wb"))






