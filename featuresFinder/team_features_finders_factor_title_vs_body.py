import sys
import re
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

import dill
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import seaborn as sns
import random
from collections import Counter
import math
import xgboost as xgb
import lightgbm as lgb

# ----------------- helpers -------------------- #
english_stemmer = nltk.stem.SnowballStemmer('english')
token_pattern = r"(?u)\b\w\w+\b"
stopwords = set(nltk.corpus.stopwords.words('english'))


def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

def preprocess_data(line,
                    token_pattern=token_pattern,
                    exclude_stopword=True,
                    stem=True):
    # token_pattern = re.compile(token_pattern, flags = re.UNICODE | re.LOCALE)
    token_pattern = re.compile(token_pattern, flags = re.UNICODE)
    tokens = [x.lower() for x in token_pattern.findall(line)]
    tokens_stemmed = tokens
    if stem:
        tokens_stemmed = stem_tokens(tokens, english_stemmer)
    if exclude_stopword:
        tokens_stemmed = [x for x in tokens_stemmed if x not in stopwords]

    return tokens_stemmed

def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val


def cosine_sim(x, y):
    try:
        if type(x) is np.ndarray: x = x.reshape(1, -1) # get rid of the warning
        if type(y) is np.ndarray: y = y.reshape(1, -1)
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        #print(x)
        #print(y)
        d = 0.
    return d

# ----------------- ngrams ------------------ #
def getUnigram(words):
    """
        Input: a list of words, e.g., ['I', 'am', 'Denny']
        Output: a list of unigram
    """
    assert type(words) == list
    return words
    
def getBigram(words, join_string, skip=0):
  """
     Input: a list of words, e.g., ['I', 'am', 'Denny']
     Output: a list of bigram, e.g., ['I_am', 'am_Denny']
     I use _ as join_string for this example.
  """
  assert type(words) == list
  L = len(words)
        #print words
  if L > 1:
    lst = []
    for i in range(L-1):
      for k in range(1,skip+2):
        if i+k < L:
          lst.append( join_string.join([words[i], words[i+k]]) )
  else:
    # set it as unigram
    lst = getUnigram(words)
  #print 'lst returned'
  return lst
    
def getTrigram(words, join_string, skip=0):
  """
     Input: a list of words, e.g., ['I', 'am', 'Denny']
     Output: a list of trigram, e.g., ['I_am_Denny']
     I use _ as join_string for this example.
  """
  assert type(words) == list
  L = len(words)
  if L > 2:
    lst = []
    for i in range(L-2):
      for k1 in range(1,skip+2):
        for k2 in range(1,skip+2):
          if i+k1 < L and i+k1+k2 < L:
            lst.append( join_string.join([words[i], words[i+k1], words[i+k1+k2]]) )
  else:
    # set it as bigram
    lst = getBigram(words, join_string, skip)
  return lst

# ----------------- score ------------------ #
LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
LABELS_RELATED = ['unrelated','related']
RELATED = LABELS[0:3]

def score_submission(gold_labels, test_labels):
    score = 0.0
    cm = [[0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]]

    for i, (g, t) in enumerate(zip(gold_labels, test_labels)):
        g_stance, t_stance = g, t
        if g_stance == t_stance:
            score += 0.25
            if g_stance != 'unrelated':
                score += 0.50
        if g_stance in RELATED and t_stance in RELATED:
            score += 0.25

        cm[LABELS.index(g_stance)][LABELS.index(t_stance)] += 1

    return score, cm


def print_confusion_matrix(cm):
    lines = []
    header = "|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format('', *LABELS)
    line_len = len(header)
    lines.append("-"*line_len)
    lines.append(header)
    lines.append("-"*line_len)

    hit = 0
    total = 0
    for i, row in enumerate(cm):
        hit += row[i]
        total += sum(row)
        lines.append("|{:^11}|{:^11}|{:^11}|{:^11}|{:^11}|".format(LABELS[i],
                                                                   *row))
        lines.append("-"*line_len)
    #print('\n'.join(lines))


def report_score(actual,predicted):
    score,cm = score_submission(actual,predicted)
    best_score, _ = score_submission(actual,actual)

    print_confusion_matrix(cm)
    #print("Score: " +str(score) + " out of " + str(best_score) + "\t("+str(score*100/best_score) + "%)")
    return score*100/best_score

# ----------------- FeatureGenerator ------------------ #
class FeatureGenerator(object):
    def __init__(self, name):
        self._name = name
    
    def name(self):
        return self._name

    def process(self, data, header):
        '''
            input:
                data: pandas dataframe
            generate features and save them into a pickle file
        '''
        pass

    def read(self, header):
        '''
            read the feature matrix from a pickle file
        '''
        pass

# ----------------- CountFeatureGenerator ------------------ #
# from FeatureGenerator import *
from nltk.tokenize import sent_tokenize
# from helpers import *
import hashlib

class CountFeatureGenerator(FeatureGenerator):
    def __init__(self, name='countFeatureGenerator'):
        super(CountFeatureGenerator, self).__init__(name)


    def process(self, df, save_file=True, test_only=False):

        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["Headline", "articleBody"]
        #print("generate counting features")
        for feat_name in feat_names:
            for gram in grams:
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
		            list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    list(map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)]))

        # overlapping n-grams count
        for gram in grams:
            df["count_of_Headline_%s_in_articleBody" % gram] = \
                list(df.apply(lambda x: sum([1. for w in x["Headline_" + gram] if w in set(x["articleBody_" + gram])]), axis=1))
            df["ratio_of_Headline_%s_in_articleBody" % gram] = \
                list(map(try_divide, df["count_of_Headline_%s_in_articleBody" % gram], df["count_of_Headline_%s" % gram]))
        
        # number of sentences in headline and body
        for feat_name in feat_names:
            #df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x.decode('utf-8').encode('ascii', errors='ignore'))))
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))
            #print df['len_sent_%s' % feat_name]

        # dump the basic counting features into a file
        feat_names = [ n for n in df.columns \
                if "count" in n \
                or "ratio" in n \
                or "len_sent" in n]
        
        # binary refuting features
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        _hedging_seed_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]
        
        #df['refuting_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #df['hedging_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #check_words = _refuting_words + _hedging_seed_words
        check_words = _refuting_words
        for rf in check_words:
            fname = '%s_exist' % rf
            feat_names.append(fname)
            df[fname] = list(df['Headline'].map(lambda x: 1 if rf in x else 0))
	    
        # number of body texts paired up with the same headline
        #df['headline_hash'] = df['Headline'].map(lambda x: hashlib.md5(x).hexdigest())
        #nb_dict = df.groupby(['headline_hash'])['Body ID'].nunique().to_dict()
        #df['n_bodies'] = df['headline_hash'].map(lambda x: nb_dict[x])
        #feat_names.append('n_bodies')
        # number of headlines paired up with the same body text
        #nh_dict = df.groupby(['Body ID'])['headline_hash'].nunique().to_dict()
        #df['n_headlines'] = df['Body ID'].map(lambda x: nh_dict[x])
        #feat_names.append('n_headlines')
        #print('BasicCountFeatures:')
        #print(df)

        if test_only:
            #print('test_only:', test_only)
            # #print(train[['Headline_unigram','Body ID', 'count_of_Headline_unigram']])
            #print('saving df cols', df.columns)
            #print('saving df head', df.head(2))
            xBasicCountsTrain = df[feat_names].values
            #print('xBasicCountsTrain shape', xBasicCountsTrain.shape)
            return [xBasicCountsTrain]
            
        
        # split into train, test portion and save in separate files
        train = df[~df['target'].isnull()]
        #print('train:')
        #print(train[['Headline_unigram','Body ID', 'count_of_Headline_unigram']])
        xBasicCountsTrain = train[feat_names].values
        #print('xBasicCountsTrain shape', xBasicCountsTrain.shape)
        outfilename_bcf_train = "train.basic.pkl"
        if save_file:
            with open(outfilename_bcf_train, "wb") as outfile:
                pickle.dump(feat_names, outfile, -1)
                pickle.dump(xBasicCountsTrain, outfile, -1)
            #print('basic counting features for training saved in %s' % outfilename_bcf_train)
        
        test = df[df['target'].isnull()]
        #print('test:')
        #print(test[['Headline_unigram','Body ID', 'count_of_Headline_unigram']])
        #return 1
        if test.shape[0] > 0:
            # test set exists
            if save_file:
                #print('saving test set')
                xBasicCountsTest = test[feat_names].values
                outfilename_bcf_test = "test.basic.pkl"
                with open(outfilename_bcf_test, 'wb') as outfile:
                    pickle.dump(feat_names, outfile, -1)
                    pickle.dump(xBasicCountsTest, outfile, -1)
                    #print('basic counting features for test saved in %s' % outfilename_bcf_test)
                    #print('xBasicCountsTest shape', xBasicCountsTest.shape)

        return [xBasicCountsTest]


    def read(self, header='train'):

        filename_bcf = "%s.basic.pkl" % header
        with open(fakenews_path + filename_bcf, "rb") as infile:
            feat_names = pickle.load(infile)
            xBasicCounts = pickle.load(infile)
            #print('feature names: ')
            #print(feat_names)
            #print('xBasicCounts.shape:')
            #print(xBasicCounts.shape)
            #print type(xBasicCounts)
            np.save('counts_test', [xBasicCounts])

        return [xBasicCounts]


# ----------------- TfidfFeatureGenerator ------------------ #
# from FeatureGenerator import *
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFeatureGenerator(FeatureGenerator):
    def __init__(self, name='tfidfFeatureGenerator'):
        super(TfidfFeatureGenerator, self).__init__(name)
    
    def process(self, df, save_file=True, test_only=False):
        
       # 1). create strings based on ' '.join(Headline_unigram + articleBody_unigram) [ already stemmed ]
        def cat_text(x):
            res = '%s %s' % (' '.join(x['Headline_unigram']), ' '.join(x['articleBody_unigram']))
            return res
        df["all_text"] = list(df.apply(cat_text, axis=1))

        if test_only:
            n_train = df.shape[0]
            #print('tfidf, n_train:',n_train)
            n_test = 0
            #print('tfidf, n_test:',n_test)
        else:
            n_train = df[~df['target'].isnull()].shape[0]
            #print('tfidf, n_train:',n_train)
            n_test = df[df['target'].isnull()].shape[0]
            #print('tfidf, n_test:',n_test)

        # 2). fit a TfidfVectorizer on the concatenated strings
        # 3). sepatately transform ' '.join(Headline_unigram) and ' '.join(articleBody_unigram)
        vec = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2)
        if test_only:
            vec = TfidfVectorizer(ngram_range=(1, 3), max_df=1, min_df=0)
        vec.fit(df["all_text"]) # Tf-idf calculated on the combined training + test set
        vocabulary = vec.vocabulary_

        vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        if test_only:
            vecH = TfidfVectorizer(ngram_range=(1, 3), max_df=1, min_df=0, vocabulary=vocabulary)
        xHeadlineTfidf = vecH.fit_transform(df['Headline_unigram'].map(lambda x: ' '.join(x))) # use ' '.join(Headline_unigram) instead of Headline since the former is already stemmed
        #print('xHeadlineTfidf.shape:')
        #print(xHeadlineTfidf.shape)
        
        # save train and test into separate files
        xHeadlineTfidfTrain = xHeadlineTfidf[:n_train, :]
        if save_file:
            outfilename_htfidf_train = "train.headline.tfidf.pkl"
            with open(outfilename_htfidf_train, "wb") as outfile:
                pickle.dump(xHeadlineTfidfTrain, outfile, -1)
            #print('headline tfidf features of training set saved in %s' % outfilename_htfidf_train)
        
        if n_test > 0:
            # test set is available
            xHeadlineTfidfTest = xHeadlineTfidf[n_train:, :]
            if save_file:
                outfilename_htfidf_test = "test.headline.tfidf.pkl"
                with open(outfilename_htfidf_test, "wb") as outfile:
                    pickle.dump(xHeadlineTfidfTest, outfile, -1)
                #print('headline tfidf features of test set saved in %s' % outfilename_htfidf_test)


        vecB = TfidfVectorizer(ngram_range=(1, 3), max_df=0.8, min_df=2, vocabulary=vocabulary)
        xBodyTfidf = vecB.fit_transform(df['articleBody_unigram'].map(lambda x: ' '.join(x)))
        #print('xBodyTfidf.shape:')
        #print(xBodyTfidf.shape)
        
        # save train and test into separate files
        xBodyTfidfTrain = xBodyTfidf[:n_train, :]
        if save_file:
            outfilename_btfidf_train = "train.body.tfidf.pkl"
            with open(outfilename_btfidf_train, "wb") as outfile:
                pickle.dump(xBodyTfidfTrain, outfile, -1)
            #print('body tfidf features of training set saved in %s' % outfilename_btfidf_train)
        
        if n_test > 0:
            # test set is availble
            xBodyTfidfTest = xBodyTfidf[n_train:, :]
            if save_file:
                outfilename_btfidf_test = "test.body.tfidf.pkl"
                with open(outfilename_btfidf_test, "wb") as outfile:
                    pickle.dump(xBodyTfidfTest, outfile, -1)
                #print('body tfidf features of test set saved in %s' % outfilename_btfidf_test)
               

        # 4). compute cosine similarity between headline tfidf features and body tfidf features
        # simTfidf = np.asarray(map(cosine_sim, xHeadlineTfidf, xBodyTfidf))[:, np.newaxis]
        # work-around for array indice error
        res = []
        if test_only:
          for i in range(0, 1):
              res.append(cosine_sim(xHeadlineTfidf[i], xBodyTfidf[i]))
        else:
          for i in range(0, 75385):
            res.append(cosine_sim(xHeadlineTfidf[i], xBodyTfidf[i]))
            
        simTfidf = np.asarray(list(res))[:, np.newaxis]
        #

        #print('simTfidf.shape:')
        #print(simTfidf.shape)
        simTfidfTrain = simTfidf[:n_train]
        if save_file:
            outfilename_simtfidf_train = "train.sim.tfidf.pkl"
            with open(outfilename_simtfidf_train, "wb") as outfile:
                pickle.dump(simTfidfTrain, outfile, -1)
            #print('tfidf sim. features of training set saved in %s' % outfilename_simtfidf_train)
        
        if n_test > 0:
            # test set is available
            simTfidfTest = simTfidf[n_train:]
            if save_file:
                outfilename_simtfidf_test = "test.sim.tfidf.pkl"
                with open(outfilename_simtfidf_test, "wb") as outfile:
                    pickle.dump(simTfidfTest, outfile, -1)
                #print('tfidf sim. features of test set saved in %s' % outfilename_simtfidf_test)

        #print('return tfidf shapes')
        #print('xHeadlineTfidf.shape:')
        #print(xHeadlineTfidf.shape)
        #print('xBodyTfidf.shape:')
        #print(xBodyTfidf.shape)
        #print('simTfidf.shape:')
        #print(simTfidf.shape)
        return [xHeadlineTfidf, xBodyTfidf, simTfidf]
        # return [simTfidf.reshape(-1, 1)]

    def read(self, header = 'train'):
        filename_htfidf = "%s.headline.tfidf.pkl" % header
        with open(fakenews_path + filename_htfidf, "rb") as infile:
            xHeadlineTfidf = pickle.load(infile)

        filename_btfidf = "%s.body.tfidf.pkl" % header
        with open(fakenews_path + filename_btfidf, "rb") as infile:
            xBodyTfidf = pickle.load(infile)

        filename_simtfidf = "%s.sim.tfidf.pkl" % header
        with open(fakenews_path + filename_simtfidf, "rb") as infile:
            simTfidf = pickle.load(infile)

        #print('xHeadlineTfidf.shape:')
        #print(xHeadlineTfidf.shape)
        #print type(xHeadlineTfidf)
        #print('xBodyTfidf.shape:')
        #print(xBodyTfidf.shape)
        #print type(xBodyTfidf)
        #print('simTfidf.shape:')
        #print(simTfidf.shape)
        #print type(simTfidf)

        # return [xHeadlineTfidf, xBodyTfidf, simTfidf.reshape(-1, 1)]
        return [simTfidf.reshape(-1, 1)]

# ----------------- SvdFeatureGenerator ------------------ #
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD

class SvdFeatureGenerator(FeatureGenerator):
    def __init__(self, name='svdFeatureGenerator'):
        super(SvdFeatureGenerator, self).__init__(name)


    def process(self, df, xHeadlineTfidfTrain=None, xBodyTfidfTrain=None, save_file=True, test_only=False):
        
        if test_only:
            n_train = df.shape[0]
            #print('SvdFeatureGenerator, n_train:',n_train)
            n_test = 0
            #print('SvdFeatureGenerator, n_test:',n_test)
        else:
            n_train = df[~df['target'].isnull()].shape[0]
            #print('SvdFeatureGenerator, n_train:',n_train)
            n_test  = df[df['target'].isnull()].shape[0]
            #print('SvdFeatureGenerator, n_test:',n_test)


        if xHeadlineTfidfTrain is not None and xBodyTfidfTrain is not None:
            #print('xHeadlineTfidfTrain not None and xBodyTfidfTrain not None')
            print("")
        else:
            #print('xHeadlineTfidfTrain is None or xBodyTfidfTrain is None')
            tfidfGenerator = TfidfFeatureGenerator('tfidf')
            featuresTrain = tfidfGenerator.read('train')
            xHeadlineTfidfTrain, xBodyTfidfTrain = featuresTrain[0], featuresTrain[1]
        
        xHeadlineTfidf = xHeadlineTfidfTrain
        xBodyTfidf = xBodyTfidfTrain
        if n_test > 0:
            # test set is available
            featuresTest  = tfidfGenerator.read('test')
            xHeadlineTfidfTest,  xBodyTfidfTest  = featuresTest[0],  featuresTest[1]
            xHeadlineTfidf = vstack([xHeadlineTfidfTrain, xHeadlineTfidfTest])
            xBodyTfidf = vstack([xBodyTfidfTrain, xBodyTfidfTest])
	    
        # compute the cosine similarity between truncated-svd features
        # svd = TruncatedSVD(n_components=50, n_iter=15)
        svd = TruncatedSVD(n_components=15, n_iter=15)
        xHBTfidf = vstack([xHeadlineTfidf, xBodyTfidf])
        svd.fit(xHBTfidf) # fit to the combined train-test set (or the full training set for cv process)
        #print('xHeadlineTfidf.shape:')
        #print(xHeadlineTfidf.shape)
        xHeadlineSvd = svd.transform(xHeadlineTfidf)
        #print('svd after transform xHeadlineSvd.shape:')
        #print(xHeadlineSvd.shape)
        
        xHeadlineSvdTrain = xHeadlineSvd[:n_train, :]
        if save_file:
            outfilename_hsvd_train = "train.headline.svd.pkl"
            with open(outfilename_hsvd_train, "wb") as outfile:
                pickle.dump(xHeadlineSvdTrain, outfile, -1)
            #print('headline svd features of training set saved in %s' % outfilename_hsvd_train)
        
        if n_test > 0:
            # test set is available
            xHeadlineSvdTest = xHeadlineSvd[n_train:, :]
            if save_file:
                outfilename_hsvd_test = "test.headline.svd.pkl"
                with open(outfilename_hsvd_test, "wb") as outfile:
                    pickle.dump(xHeadlineSvdTest, outfile, -1)
                #print('headline svd features of test set saved in %s' % outfilename_hsvd_test)

        xBodySvd = svd.transform(xBodyTfidf)
        #print('xBodySvd.shape:')
        #print(xBodySvd.shape)
        
        xBodySvdTrain = xBodySvd[:n_train, :]
        if save_file:
            outfilename_bsvd_train = "train.body.svd.pkl"
            with open(outfilename_bsvd_train, "wb") as outfile:
                pickle.dump(xBodySvdTrain, outfile, -1)
            #print('body svd features of training set saved in %s' % outfilename_bsvd_train)
        
        if n_test > 0:
            # test set is available
            xBodySvdTest = xBodySvd[n_train:, :]
            if save_file:
                outfilename_bsvd_test = "test.body.svd.pkl"
                with open(outfilename_bsvd_test, "wb") as outfile:
                    pickle.dump(xBodySvdTest, outfile, -1)
                #print('body svd features of test set saved in %s' % outfilename_bsvd_test)

        # work-around for array indice error
        # simSvd = np.asarray(map(cosine_sim, xHeadlineSvd, xBodySvd))[:, np.newaxis]
        res = []
        if test_only:
          for i in range(0, 1):
            res.append(cosine_sim(xHeadlineSvd[i], xBodySvd[i]))
        else:
          for i in range(0, 75385):
            res.append(cosine_sim(xHeadlineSvd[i], xBodySvd[i]))

        simSvd = np.asarray(list(res))[:, np.newaxis]
        
        #print('simSvd.shape:')
        #print(simSvd.shape)

        simSvdTrain = simSvd[:n_train]
        if save_file:
            outfilename_simsvd_train = "train.sim.svd.pkl"
            with open(outfilename_simsvd_train, "wb") as outfile:
                pickle.dump(simSvdTrain, outfile, -1)
            #print('svd sim. features of training set saved in %s' % outfilename_simsvd_train)
        
        if n_test > 0:
            # test set is available
            simSvdTest = simSvd[n_train:]
            outfilename_simsvd_test = "test.sim.svd.pkl"
            if save_file:
                with open(outfilename_simsvd_test, "wb") as outfile:
                    pickle.dump(simSvdTest, outfile, -1)
                #print('svd sim. features of test set saved in %s' % outfilename_simsvd_test)

        if test_only:
            # pad with 0 since the model has much more values after truncated
            #print('manually add empty features')
            #print('xHeadlineSvd mine:', xHeadlineSvd)

            #print('xHeadlineSvd mine 0:',xHeadlineSvd[0])
            #print('xHeadlineSvd mine 00:',xHeadlineSvd[0][0])

            result1 = np.pad(xHeadlineSvd[0], (0, 48), 'constant') # pad with 0s, 0 offset left, 48 right
            xHeadlineSvd = [result1]
            # array([0, 0, 1, 2, 3, 4, 5, 0, 0, 0])
            result2 = np.pad(xBodySvd[0], (0, 48), 'constant') # pad with 0s, 0 offset left, 48 right
            xBodySvd = [result2]
            
            #print('return svd shapes')
            #print('xHeadlineSvd.shape:')
            # #print(xHeadlineSvd.shape)
            #print('xBodySvd.shape:')
            # #print(xBodySvd.shape)
            #print('simSvd.shape:')
            #print(type(simSvd))
            # #print(simSvd.shape)

            return [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]


        return [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]


    def read(self, header='train'):

        filename_hsvd = "%s.headline.svd.pkl" % header
        with open(fakenews_path + filename_hsvd, "rb") as infile:
            xHeadlineSvd = pickle.load(infile)

        filename_bsvd = "%s.body.svd.pkl" % header
        with open(fakenews_path + filename_bsvd, "rb") as infile:
            xBodySvd = pickle.load(infile)

        filename_simsvd = "%s.sim.svd.pkl" % header
        with open(fakenews_path + filename_simsvd, "rb") as infile:
            simSvd = pickle.load(infile)

        #print('xHeadlineSvd.shape:')
        #print(xHeadlineSvd.shape)
        ##print(type(xHeadlineSvd))
        #print('xBodySvd.shape:')
        #print(xBodySvd.shape)
        ##print(type(xBodySvd))
        #print('simSvd.shape:')
        #print(simSvd.shape)
        ##print(type(simSvd))

        return [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]
        #return [simSvd.reshape(-1, 1)]

# ----------------- Word2VecFeatureGenerator ------------------ #
import gensim
from sklearn.preprocessing import normalize
from functools import reduce

class Word2VecFeatureGenerator(FeatureGenerator):
    model = gensim.models.KeyedVectors.load_word2vec_format("/content/drive/My Drive/MLFall2020/the-feature-finders/datasets/fakenewschallenge/GoogleNews-vectors-negative300.bin", binary=True)

    def __init__(self, name='word2vecFeatureGenerator'):
        super(Word2VecFeatureGenerator, self).__init__(name)

    def process(self, df, save_file=True, test_only=False):

        #print('generating word2vec features')
        df["Headline_unigram_vec"] = df["Headline"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
        df["articleBody_unigram_vec"] = df["articleBody"].map(lambda x: preprocess_data(x, exclude_stopword=False, stem=False))
        
        if test_only:
          n_train = df.shape[0]
          #print('Word2VecFeatureGenerator: n_train:',n_train)
          n_test = 0
          #print('Word2VecFeatureGenerator: n_test:',n_test)
        else:
          n_train = df[~df['target'].isnull()].shape[0]
          #print('Word2VecFeatureGenerator: n_train:',n_train)
          n_test = df[df['target'].isnull()].shape[0]
          #print('Word2VecFeatureGenerator: n_test:',n_test)
        
        # 1). document vector built by multiplying together all the word vectors
        # using Google's pre-trained word vectors
        # model = gensim.models.KeyedVectors.load_word2vec_format("/content/drive/My Drive/mydata/fakenewschallenge/GoogleNews-vectors-negative300.bin", binary=True)
        #print('google news model loaded')

        Headline_unigram_array = df['Headline_unigram_vec'].values
        #print('Headline_unigram_array:')
        #print(Headline_unigram_array)
        #print(Headline_unigram_array.shape)
        #print(type(Headline_unigram_array))
        
        # word vectors weighted by normalized tf-idf coefficient?
        #headlineVec = [0]
        headlineVec = list(map(lambda x: reduce(np.add, [Word2VecFeatureGenerator.model[y] for y in x if y in Word2VecFeatureGenerator.model], [0.]*300), Headline_unigram_array))
        headlineVec = np.array(headlineVec)
        #print('headlineVec:')
        #print(headlineVec)
        #print('type(headlineVec)')
        #print(type(headlineVec))
        #headlineVec = np.exp(headlineVec)
        headlineVec = normalize(headlineVec)
        #print('headlineVec')
        #print(headlineVec)
        #print(headlineVec.shape)
        
        headlineVecTrain = headlineVec[:n_train, :]
        if save_file:
            outfilename_hvec_train = "train.headline.word2vec.pkl"
            with open(outfilename_hvec_train, "wb") as outfile:
                pickle.dump(headlineVecTrain, outfile, -1)
            #print('headline word2vec features of training set saved in %s' % outfilename_hvec_train)

        if n_test > 0:
            # test set is available
            headlineVecTest = headlineVec[n_train:, :]
            if save_file:
                outfilename_hvec_test = "test.headline.word2vec.pkl"
                with open(outfilename_hvec_test, "wb") as outfile:
                    pickle.dump(headlineVecTest, outfile, -1)
                #print('headline word2vec features of test set saved in %s' % outfilename_hvec_test)
        #print('headine done')

        Body_unigram_array = df['articleBody_unigram_vec'].values
        #print('Body_unigram_array:')
        #print(Body_unigram_array)
        #print(Body_unigram_array.shape)
        #bodyVec = [0]
        bodyVec = list(map(lambda x: reduce(np.add, [Word2VecFeatureGenerator.model[y] for y in x if y in Word2VecFeatureGenerator.model], [0.]*300), Body_unigram_array))
        bodyVec = np.array(bodyVec)
        bodyVec = normalize(bodyVec)
        #print('bodyVec')
        #print(bodyVec)
        #print(bodyVec.shape)

        bodyVecTrain = bodyVec[:n_train, :]
        if save_file:
            outfilename_bvec_train = "train.body.word2vec.pkl"
            with open(outfilename_bvec_train, "wb") as outfile:
                pickle.dump(bodyVecTrain, outfile, -1)
            #print('body word2vec features of training set saved in %s' % outfilename_bvec_train)
        
        if n_test > 0:
            # test set is available
            bodyVecTest = bodyVec[n_train:, :]
            if save_file:
                outfilename_bvec_test = "test.body.word2vec.pkl"
                with open(outfilename_bvec_test, "wb") as outfile:
                    pickle.dump(bodyVecTest, outfile, -1)
                #print('body word2vec features of test set saved in %s' % outfilename_bvec_test)

        #print('body done')

        # compute cosine similarity between headline/body word2vec features
        # simVec = np.asarray(map(cosine_sim, headlineVec, bodyVec))[:, np.newaxis]
        # work-around for indice error
        res = []
        if test_only:
            for i in range(0, 1):
                res.append(cosine_sim(headlineVec[i], bodyVec[i]))
        else:
            for i in range(0, 75385):
                res.append(cosine_sim(headlineVec[i], bodyVec[i]))
        simVec = np.asarray(list(res))[:, np.newaxis]
        #print('simVec.shape:')
        #print(simVec.shape)

        simVecTrain = simVec[:n_train]
        if save_file:
            outfilename_simvec_train = "train.sim.word2vec.pkl"
            with open(outfilename_simvec_train, "wb") as outfile:
                pickle.dump(simVecTrain, outfile, -1)
            #print('word2vec sim. features of training set saved in %s' % outfilename_simvec_train)
        
        if n_test > 0:
            # test set is available
            simVecTest = simVec[n_train:]
            if save_file:
                outfilename_simvec_test = "test.sim.word2vec.pkl"
                with open(outfilename_simvec_test, "wb") as outfile:
                    pickle.dump(simVecTest, outfile, -1)
                #print('word2vec sim. features of test set saved in %s' % outfilename_simvec_test)

        #print('return w2vec shapes')
        #print('headlineVecTrain.shape:')
        #print(headlineVecTrain.shape)
        #print('bodyVecTrain.shape:')
        #print(bodyVecTrain.shape)
        #print('simVecTrain.shape:')
        #print(simVecTrain.shape)
        return [headlineVecTrain, bodyVecTrain, simVecTrain]

    def read(self, header='train'):

        filename_hvec = "%s.headline.word2vec.pkl" % header
        with open(fakenews_path + filename_hvec, "rb") as infile:
            headlineVec = pickle.load(infile)

        filename_bvec = "%s.body.word2vec.pkl" % header
        with open(fakenews_path + filename_bvec, "rb") as infile:
            bodyVec = pickle.load(infile)

        filename_simvec = "%s.sim.word2vec.pkl" % header
        with open(fakenews_path + filename_simvec, "rb") as infile:
            simVec = pickle.load(infile)

        #print('headlineVec.shape:')
        #print(headlineVec.shape)
        #print('bodyVec.shape:')
        #print(bodyVec.shape)
        #print('simVec.shape:')
        #print(simVec.shape)

        return [headlineVec, bodyVec, simVec]
        #return [simVec.reshape(-1,1)]

# ----------------- SentimentFeatureGenerator ------------------ #
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize

class SentimentFeatureGenerator(FeatureGenerator):
    def __init__(self, name='sentimentFeatureGenerator'):
        super(SentimentFeatureGenerator, self).__init__(name)


    def process(self, df, save_file=True, test_only=False):

        #print('generating sentiment features')
        #print('for headline')
        
        if test_only:
          n_train = df.shape[0]
          #print('SentimentFeatureGenerator: n_train:',n_train)
          n_test = 0
          #print('SentimentFeatureGenerator: n_test:',n_test)
        else:
          n_train = df[~df['target'].isnull()].shape[0]
          #print('SentimentFeatureGenerator: n_train:',n_train)
          n_test = df[df['target'].isnull()].shape[0]
          #print('SentimentFeatureGenerator: n_test:',n_test)

        # calculate the polarity score of each sentence then take the average
        sid = SentimentIntensityAnalyzer()
        def compute_sentiment(sentences):
            result = []
            for sentence in sentences:
                vs = sid.polarity_scores(sentence)
                result.append(vs)
            return pd.DataFrame(result).mean()
        
        #df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x.decode('utf-8')))
        df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'h_compound', 'neg':'h_neg', 'neu':'h_neu', 'pos':'h_pos'}, inplace=True)
        #print 'df:'
        #print df
        #print df.columns
        #print df.shape
        headlineSenti = df[['h_compound','h_neg','h_neu','h_pos']].values
        #print('headlineSenti.shape:')
        #print(headlineSenti.shape)
        
        headlineSentiTrain = headlineSenti[:n_train, :]
        if save_file:
            outfilename_hsenti_train = "train.headline.senti.pkl"
            with open(outfilename_hsenti_train, "wb") as outfile:
                pickle.dump(headlineSentiTrain, outfile, -1)
            #print('headline sentiment features of training set saved in %s' % outfilename_hsenti_train)
        
        if n_test > 0:
            # test set is available
            headlineSentiTest = headlineSenti[n_train:, :]
            if save_file:
                outfilename_hsenti_test = "test.headline.senti.pkl"
                with open(outfilename_hsenti_test, "wb") as outfile:
                    pickle.dump(headlineSentiTest, outfile, -1)
                #print('headline sentiment features of test set saved in %s' % outfilename_hsenti_test)
        
        #print('headine senti done')
        
        #return 1

        #print('for body')
        #df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x.decode('utf-8')))
        df['body_sents'] = df['articleBody'].map(lambda x: sent_tokenize(x))
        df = pd.concat([df, df['body_sents'].apply(lambda x: compute_sentiment(x))], axis=1)
        df.rename(columns={'compound':'b_compound', 'neg':'b_neg', 'neu':'b_neu', 'pos':'b_pos'}, inplace=True)
        #print 'body df:'
        #print df
        #print df.columns
        bodySenti = df[['b_compound','b_neg','b_neu','b_pos']].values
        #print('bodySenti.shape:')
        #print(bodySenti.shape)
        
        bodySentiTrain = bodySenti[:n_train, :]
        if save_file:
            outfilename_bsenti_train = "train.body.senti.pkl"
            with open(outfilename_bsenti_train, "wb") as outfile:
                pickle.dump(bodySentiTrain, outfile, -1)
            #print('body sentiment features of training set saved in %s' % outfilename_bsenti_train)
        
        if n_test > 0:
            # test set is available
            bodySentiTest = bodySenti[n_train:, :]
            if save_file:
                outfilename_bsenti_test = "test.body.senti.pkl"
                with open(outfilename_bsenti_test, "wb") as outfile:
                    pickle.dump(bodySentiTest, outfile, -1)
                #print('body sentiment features of test set saved in %s' % outfilename_bsenti_test)

        #print('body senti done')

        #print('senti return shapes')
        #print('headlineSentiTrain.shape:')
        #print(headlineSentiTrain.shape)
        #print('bodySentiTrain.shape:')
        #print(bodySentiTrain.shape)

        return [headlineSentiTrain, bodySentiTrain]


    def read(self, header='train'):

        filename_hsenti = "%s.headline.senti.pkl" % header
        with open(fakenews_path + filename_hsenti, "rb") as infile:
            headlineSenti = pickle.load(infile)

        filename_bsenti = "%s.body.senti.pkl" % header
        with open(fakenews_path + filename_bsenti, "rb") as infile:
            bodySenti = pickle.load(infile)
        np.save('senti_headline_body_test', [headlineSenti, bodySenti])
        #print('headlineSenti.shape:')
        #print(headlineSenti.shape)
        #print type(headlineSenti)
        #print('bodySenti.shape:')
        #print(bodySenti.shape)
        #print type(bodySenti)

        return [headlineSenti, bodySenti]

# ----------------- TitleVsBody ------------------ #
# imports
# team_features-finders-factor_title_vs_body.py
# from team_features_finders_factor_title_vs_body import TitleVsBody as tvb

import random
from collections import Counter
import math
import xgboost as xgb
import lightgbm as lgb

class TitleVsBody():
    model_path = "/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/"
    file_name = "titlevsbody_logistic_proba.pkl"
    titlevsbody_logistic_proba = pickle.load(open(model_path + file_name, "rb"))

    '''
    Return a list with a single predicted value (int 0-3) for Title Vs Body 
    '''
    def FeatureFinders_getTitleVsBodyRelationship(self, head_line="", body_text=""):
        #print("head_line=", head_line)
        #print("body_text=", body_text)
        head_line = head_line.strip()
        body_text = body_text.strip()

        if not head_line or not body_text: # return 3-unrelated if BAD data!!!
            print("Bad body data")
            return [3]

        init_data = { 'Headline': [head_line], 'articleBody': [body_text] }

        data = pd.DataFrame(init_data, columns = ['Headline', 'articleBody'])
        #print(data.head())

        #print('Generate unigrams', data.shape)
        # generate unigram
        data["Headline_unigram"] = data["Headline"].map(lambda x: preprocess_data(x))
        data["articleBody_unigram"] = data["articleBody"].map(lambda x: preprocess_data(x))

        # generate bigram
        ##print('Generate bigrams', data.shape)
        join_str = "_"
        data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: getBigram(x, join_str))
        data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: getBigram(x, join_str))

        # generate trigram
        #print('Generate trigrams', data.shape)
        data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: getTrigram(x, join_str))
        data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: getTrigram(x, join_str))
        
        #print('shape:', data.shape)
        #print('columns:', data.columns)
        #print('after', data.head())

        c_fg = CountFeatureGenerator()
        basic_count = c_fg.process(data, save_file=False, test_only=True)
        #print(type(basic_count))

        tfidf_fg = TfidfFeatureGenerator()
        # tfidf = [xHeadlineTfidf, xBodyTfidf, simTfidf]
        tfidf = tfidf_fg.process(data, save_file=False, test_only=True)
        #print("tfidf:", tfidf)
        #print(type(tfidf))
        
        # tfidfGenerator = TfidfFeatureGenerator('tfidf')
        # featuresTrain = tfidfGenerator.read('train')
        xHeadlineTfidfTrain, xBodyTfidfTrain, simTfidf = tfidf[0], tfidf[1], tfidf[2]
        simTfidf = [simTfidf.reshape(-1, 1)]
        #print("simTfidf len:", len(simTfidf))
        #print("simTfidf:", simTfidf)
        # return [simTfidf.reshape(-1, 1)]


        # [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]
        svd_fg = SvdFeatureGenerator()
        # svd = [xHeadlineSvd, xBodySvd, simSvd.reshape(-1, 1)]
        svd = svd_fg.process(data, xHeadlineTfidfTrain=xHeadlineTfidfTrain, xBodyTfidfTrain=xBodyTfidfTrain, save_file=False, test_only=True)
        #print("svd len:", len(svd))
        #print("svd:", svd)
        #print(type(svd))


        w2v_fg = Word2VecFeatureGenerator()
        # w2v = [headlineVecTrain, bodyVecTrain, simVecTrain]
        w2v = w2v_fg.process(data, save_file=False, test_only=True)
        #print("w2v len:", len(w2v))
        #print("w2v:", w2v)
        #print(type(w2v))

        senti_fg = SentimentFeatureGenerator()
        # senti = [headlineSentiTrain, bodySentiTrain]
        senti = senti_fg.process(data, save_file=False, test_only=True)
        #print("senti len:", len(senti))
        #print("senti:", senti)
        #print(type(senti))
        
        # features = [f for g in generators for f in g.read('train')]
        values = [basic_count, simTfidf, svd, w2v, senti]
        features = [f for v in values for f in v]
        # for v in values:
        #     for f in v:
        #         features.append(f)
        
        #print ((features))
        data_x = np.hstack(features)
        test_x = data_x
        #print('test_x len:')
        #print(len(test_x))
        #print(data_x[0,:])
        #print('data_x.shape')
        #print(data_x.shape)

        # dtrain = xgb.DMatrix(data_x, label=data_y, weight=w)
        dtest = xgb.DMatrix(test_x)

        # load model
        model_path = "/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/"
        file_name = "titlevsbody_xgboost.pkl"
        bst = pickle.load(open(model_path + file_name, "rb"))
        #print('bst type:', type(bst))

        pred = bst.predict(dtest)
        pred_prob_y = pred.reshape(test_x.shape[0], 4)
        pred_prob_y = bst.predict(dtest).reshape(test_x.shape[0], 4)
        #print('pred_prob_y:', pred_prob_y)

        pred_y = np.argmax(pred_prob_y, axis = 1)
        #print('pred_y.shape:', pred_y.shape)
        #print('pred_y:', pred_y)
        
        print("predicted titleVsBody:", str(pred_y))

        return pred_y

    def FeatureFinders_getTitleVsBodyScore(self, val): # int val 0-3 of TitleVsBody labels
        # model_path = "/content/drive/My Drive/the-feature-finders/AlternusVera/pickled-model/"
        # file_name = "titlevsbody_logistic_proba.pkl"
        # titlevsbody_logistic_proba = pickle.load(open(model_path + file_name, "rb"))
        arr = np.array([val])
        return TitleVsBody.titlevsbody_logistic_proba.predict_proba(arr.reshape(-1, 1))
