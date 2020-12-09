# -*- coding: utf-8 -*-
"""Untitled2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LXoUf5SoN8bwCupGwsoprMCZ0Njxn_E3
"""

import pandas as pd
import numpy as np
import re
# import sys
# import random
# import argparse
import pickle
# from collections import defaultdict
# from csv import DictReader
from tqdm import tqdm
from sklearn import feature_extraction
import statistics
from gensim.models.doc2vec import TaggedDocument
# stop words are, is, the etc. which are not needed for model
from nltk.corpus import stopwords
from nltk.corpus import words
from textblob import TextBlob, Word

import nltk
nltk.download('punkt')
nltk.download('wordnet')


class StanceDitectionFeature():

    _wnl = nltk.WordNetLemmatizer()

    def gen_or_load_feats(self,feat_fn, headlines, bodies, feature_file):
      feats = feat_fn(headlines, bodies)
      np.save(feature_file, feats)
      return np.load(feature_file)

    def normalize_word(self,w):
        return self._wnl.lemmatize(w).lower()

    def get_tokenized_lemmas(self,s):
        return [self.normalize_word(t) for t in nltk.word_tokenize(s)]

    def clean(self,s):
        # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

    def remove_stopwords(self,l):
        # Removes stopwords from a list of tokens
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

    def word_overlap_features(self,headlines, bodies):
      X = []
      for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
          clean_headline = self.clean(headline)
          clean_body = self.clean(body)
          clean_headline = self.get_tokenized_lemmas(clean_headline)
          clean_body = self.get_tokenized_lemmas(clean_body)
          features = [
              len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
          X.append(features)
          i = i+1
      return X

    def refuting_features(self,headlines, bodies):
      _refuting_words = [
          'fake',
          'fraud',
          'hoax',
          'false',
          'deny', 'denies',
          'not',
          'despite',
          'nope',
          'doubt', 'doubts',
          'bogus',
          'debunk',
          'pranks',
          'retract'
      ]
      X = []
      for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
          clean_headline = self.clean(headline)
          clean_headline = self.get_tokenized_lemmas(clean_headline)
          features = [1 if word in clean_headline else 0 for word in _refuting_words]
          X.append(features)
      return X

    def polarity_features(self,headlines, bodies):
      _refuting_words = [
          'fake',
          'fraud',
          'hoax',
          'false',
          'deny', 'denies',
          'not',
          'despite',
          'nope',
          'doubt', 'doubts',
          'bogus',
          'debunk',
          'pranks',
          'retract'
      ]

      def calculate_polarity(text):
          tokens = self.get_tokenized_lemmas(text)
          return sum([t in _refuting_words for t in tokens]) % 2
      X = []
      for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
          clean_headline = self.clean(headline)
          clean_body = self.clean(body)
          features = []
          features.append(calculate_polarity(clean_headline))
          features.append(calculate_polarity(clean_body))
          X.append(features)
      return np.array(X)

    def ngrams(self,input, n):
      input = input.split(' ')
      output = []
      for i in range(len(input) - n + 1):
          output.append(input[i:i + n])
      return output

    def chargrams(self,input, n):
        output = []
        for i in range(len(input) - n + 1):
            output.append(input[i:i + n])
        return output

    def append_chargrams(self,features, text_headline, text_body, size):
      grams = [' '.join(x) for x in self.chargrams(" ".join(self.remove_stopwords(text_headline.split())), size)]
      grams_hits = 0
      grams_early_hits = 0
      grams_first_hits = 0
      for gram in grams:
          if gram in text_body:
              grams_hits += 1
          if gram in text_body[:255]:
              grams_early_hits += 1
          if gram in text_body[:100]:
              grams_first_hits += 1
      features.append(grams_hits)
      features.append(grams_early_hits)
      features.append(grams_first_hits)
      return features

    def append_ngrams(self,features, text_headline, text_body, size):
        grams = [' '.join(x) for x in self.ngrams(text_headline, size)]
        grams_hits = 0
        grams_early_hits = 0
        for gram in grams:
            if gram in text_body:
                grams_hits += 1
            if gram in text_body[:255]:
                grams_early_hits += 1
        features.append(grams_hits)
        features.append(grams_early_hits)
        return features

    def hand_features(self,headlines, bodies):

      def binary_co_occurence(headline, body):
          # Count how many times a token in the title
          # appears in the body text.
          bin_count = 0
          bin_count_early = 0
          for headline_token in self.clean(headline).split(" "):
              if headline_token in self.clean(body):
                  bin_count += 1
              if headline_token in self.clean(body)[:255]:
                  bin_count_early += 1
          return [bin_count, bin_count_early]

      def binary_co_occurence_stops(headline, body):
          # Count how many times a token in the title
          # appears in the body text. Stopwords in the title
          # are ignored.
          bin_count = 0
          bin_count_early = 0
          for headline_token in self.remove_stopwords(self.clean(headline).split(" ")):
              if headline_token in self.clean(body):
                  bin_count += 1
                  bin_count_early += 1
          return [bin_count, bin_count_early]

      def count_grams(headline, body):
          # Count how many times an n-gram of the title
          # appears in the entire body, and intro paragraph

          clean_body = self.clean(body)
          clean_headline = self.clean(headline)
          features = []
          features = self.append_chargrams(features, clean_headline, clean_body, 2)
          features = self.append_chargrams(features, clean_headline, clean_body, 8)
          features = self.append_chargrams(features, clean_headline, clean_body, 4)
          features = self.append_chargrams(features, clean_headline, clean_body, 16)
          features = self.append_ngrams(features, clean_headline, clean_body, 2)
          features = self.append_ngrams(features, clean_headline, clean_body, 3)
          features = self.append_ngrams(features, clean_headline, clean_body, 4)
          features = self.append_ngrams(features, clean_headline, clean_body, 5)
          features = self.append_ngrams(features, clean_headline, clean_body, 6)
          return features

      X = []
      for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
          X.append(binary_co_occurence(headline, body)
                   + binary_co_occurence_stops(headline, body)
                   + count_grams(headline, body))

      return X

    ###
    LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
    stance_score = {'agree': 1.0, 'disagree': 0.3,
                    'discuss': 0.8, 'unrelated': 0.0}

    def __generate_fn_features(self, dataset, name):
        h, b = [], []
 # base_path = '/content/drive/MyDrive/MLFall2020/the-feature-finders/AlternusVera/Stance/'

        for d in dataset:
            h.append(d[0])  # title
            b.append(d[1])  # text

        X_overlap = self.gen_or_load_feats(
            self.word_overlap_features, h, b, self.base_path+"features/overlap."+name+".npy")
        X_refuting = self.gen_or_load_feats(
            self.refuting_features, h, b, self.base_path+"features/refuting."+name+".npy")
        X_polarity = self.gen_or_load_feats(
            self.polarity_features, h, b, self.base_path+"features/polarity."+name+".npy")
        X_hand = self.gen_or_load_feats(
            self.hand_features, h, b, self.base_path+"features/hand."+name+".npy")

        X = np.c_[X_hand, X_polarity, X_refuting, X_overlap]
        return X

    def __predict_stance(self, headline, body):
        fn_predicted = ''

        # creating the dataframe with our text so we can leverage the existing code
        fn_dataset = pd.DataFrame(index=[0], columns=['Statement', 'text'])
        fn_dataset['Statement'] = headline
        fn_dataset['text'] = body
        fn_dataset = fn_dataset.values

        X_fn = self.__generate_fn_features(fn_dataset, "StanceFeature")

# # Load the best classifier saved model
# readfile = open('/content/drive/MyDrive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/stance-model-GBC', 'rb')
# best_clf = pickle.load(readfile)

        self.fn_predicted = [self.LABELS[int(a)]
                             for a in self.bestModel.predict(X_fn)]

        return self.stance_score[self.fn_predicted[0]]

    def __init__(self, filenameModel, filenameBestModel, base_path):
        self.model = self.__load(filenameModel)
        self.bestModel = self.__load(filenameBestModel)
        self.base_path = base_path

    def __load(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    def FeatureFinders_getStanceScore(self, headline, body):
        x = self.__predict_stance(headline, body)
        xTrain = np.array(x).reshape(-1, 1)

        #   readfile = open('/content/drive/MyDrive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/stanceLabelGNB-model', 'rb')
        #   best_clf = pickle.load(readfile)

        #   xPpredicted = best_clf.predict(xTrain)
        #   xPredicedProb = best_clf.predict_proba(xTrain)[:,1]
        #   # print(x)
        xPpredicted = self.model.predict(xTrain)
        xPredicedProb = self.model.predict_proba(xTrain)[:, 1]

        return float(xPredicedProb)


class ReliableSource():

    def __init__(self, filenameModel, filenameData):
        # path = "/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/ReliableSource/data.csv"
        self.model = self.__load(filenameModel)
        self.data = filenameData

    def __load(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    # return between 0 and 1, being 0 = True,  1 = Fake
    def __FeatureFinders_getSourceReliabilityScore(self, source):
        #			path = "/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/ReliableSource/data.csv"
        #			fakeNewsSites = pd.read_csv(path)
        fakeNewsSites = pd.read_csv(self.data)
        for index, row in fakeNewsSites.iterrows():
            score = 100
            if (row['Type of site'] == 'Some fake stories'):
                score = 50
            fakeNewsSites.at[index, 'fake_score'] = score

        if (source == ""):
            return 0
        # print(source)
        d = fakeNewsSites[fakeNewsSites['Site name'].str.match(
            r'\b' + source + r'\b')]
        # print(d)
        if d.shape[0] > 0:
            return d.iloc[0]['fake_score']

        # if (d['fake_score'].empty):
        #     return 0
        # return int(d['fake_score'].values)
        return 0

    def FeatureFinders_getReliabilityBySource(self, src):
        x = self.__FeatureFinders_getSourceReliabilityScore(src)
        xTrain = np.array(x).reshape(-1, 1)

#			readfile = open('/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/ReliableSourceLabelmodel', 'rb')
#			best_clf = pickle.load(readfile)
#			xPpredicted = best_clf.predict(xTrain)
#			print(xPpredicted)
#			xPredicedProb = best_clf.predict_proba(xTrain)[:,1]
#			#xPredicedProb = best_clf.predict_proba(xTrain)
#			#print(xPredicedProb)

        xPpredicted = self.model.predict(xTrain)
        xPredicedProb = self.model.predict_proba(xTrain)[:, 1]

        return 1 - float(xPredicedProb)


class ToxicityFeature():

    def __init__(self, filenameModel, filenameBestModel, filenameFakeNewsmodel):
        self.model = self.__load(filenameModel)
        self.bestModel = self.__load(filenameBestModel)
        self.fakeNewsmodel = self.__load(filenameFakeNewsmodel)

    def __load(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    # Corpus cleaning
    def __clean_str(self, string):
        STOPWORDS = set(stopwords.words('english'))
        """
      Tokenization/string cleaning for datasets.
      Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
      """
        string = re.sub(r"^b", "", string)
        string = re.sub(r"\\n ", "", string)
        string = re.sub(r"\'s", "", string)
        string = re.sub(r"\'ve", "", string)
        string = re.sub(r"n\'t", "", string)
        string = re.sub(r"\'re", "", string)
        string = re.sub(r"\'d", "", string)
        string = re.sub(r"\'ll", "", string)
        string = re.sub(r",", "", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", "", string)
        string = re.sub(r"\)", "", string)
        string = re.sub(r"\?", "", string)
        string = re.sub(r"'", "", string)
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"[0-9]\w+|[0-9]", "", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = ' '.join(Word(word).lemmatize() for word in string.split(
        ) if word not in STOPWORDS)  # delete stopwors from text
        return string.strip().lower()

    def __label_sentences(self, corpus, label_type):
        """
        Gensim's Doc2Vec implementation requires each document/paragraph to have a label associated with it.
        We do this by using the TaggedDocument method. The format will be "TRAIN_i" or "TEST_i" where "i" is
        a dummy index of the post.
        """
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(TaggedDocument(v.split(), [label]))
        return labeled

    def __get_vectors(self, model, corpus_size, vectors_size, vectors_type):
        """
        Get vectors from trained doc2vec model
        :param doc2vec_model: Trained Doc2Vec model
        :param corpus_size: Size of the data
        :param vectors_size: Size of the embedding vectors
        :param vectors_type: Training or Testing vectors
        :return: list of vectors
        """
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors

    def FeatureFinders_getToxicityScore(self, headline, body):
        # clean
        headline = self.__clean_str(headline)
        body = self.__clean_str(body)

        all_text = headline+body

        # print(all_text)

        all_text = self.__label_sentences(all_text, "Test")

#        dbowfile = open(
#            '/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/model_dbow', 'rb')
#        model_dbow = pickle.load(dbowfile)

        # Doc2Vec
        test1_vectors_dbow = self.__get_vectors(
            self.model, len(all_text), 300, 'Test')

        # best Model for toxicity prediction
#        toxicityModel_file = open(
#            '/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/toxicity-model', 'rb')
#        best_clf = pickle.load(toxicityModel_file)

        # predictions
        predictedToxicity = self.bestModel.predict(test1_vectors_dbow)
        predictedToxicity = set(predictedToxicity)
        predictedToxicity = statistics.mean(predictedToxicity)

#        predictedToxicity = best_clf.predict(test1_vectors_dbow)
#        predictedToxicity = set(predictedToxicity)
#        predictedToxicity = statistics.mean(predictedToxicity)
        #predicedProb = best_clf.predict_proba(test1_vectors_dbow)[:,1]
        # print(predictedToxicity)

        # best Model for fakenews prediction
#        toxicityLabel_file = open(
#            '/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/toxicityFakenewsLabel-model', 'rb')
#        best_labelclf = pickle.load(toxicityLabel_file)

        predictedFakeNews = self.fakeNewsmodel.predict(predictedToxicity)
        predicedProb = self.fakeNewsmodel.predict_proba(predictedToxicity)[
            :, 0]

#        predictedFakeNews = best_labelclf.predict(predictedToxicity)
#        predicedProb = best_labelclf.predict_proba(predictedToxicity)[:, 0]

        # print(predictedFakeNews)
        # print(predicedProb)
        return float(predicedProb[0])
