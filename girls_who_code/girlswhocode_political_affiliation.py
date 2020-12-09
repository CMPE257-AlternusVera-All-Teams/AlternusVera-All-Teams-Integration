# -*- coding: utf-8 -*-
"""girlswhocode_political_affiliation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/174UwkYutB7AJvFMiEBY0DbC0eEiPfPOd
"""

# load required libraries and read data
import os
import pandas as pd
import numpy as np
import seaborn as sns
import gensim
import string
import operator
import re
import nltk
import math
import pickle
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import  LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim.models import word2vec
from nltk import word_tokenize
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer 
from scipy import spatial
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

#mount google drive
from google.colab import drive
drive.mount('/content/drive')

#pip install vaderSentiment

#Vader Sentiment Analyser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class Girlswhocode_PoliticalAfiiliation:
 
  nltk.download('stopwords')

  #label encoding
  def encode_news_type(input_label):
      true_labels = ['original','true','mostly-true','half-true']
      false_labels = ['barely-true','false','pants-fire']
      if input_label in true_labels:
          return 1
      else:
          return 0

  #method to remove punctuations from textual data
  def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

 

  #Remove stop words
  def remove_stopwords(text):
      sw = stopwords.words('english')
      stemmer = SnowballStemmer("english")
      text = [word.lower() for word in text.split() if word.lower() not in sw]
      return " ".join(text)

  #Lemmetize and pos tagging
  def lemmatize_stemming(text):
      sw = stopwords.words('english')
      stemmer = SnowballStemmer("english")
      return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

  #Stemming
  def stemming(text): 
      sw = stopwords.words('english')
      stemmer = SnowballStemmer("english")
      text = [stemmer.stem(word) for word in text.split()]
      return " ".join(text)

  def text_preprocess(df):
    #encode labels
    #df['encoded_label'] = df.apply(lambda row: encode_news_type(row['label']), axis=1)
    #convert to lower case
    df['headline_text'] = df['headline_text'].str.lower()
    #remove stop words
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.remove_stopwords)
    #spell check
    #df['headline_text'] = df['headline_text'].apply(spell_checker)
    #Lemmetize
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.lemmatize_stemming)
    #stemming
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.stemming)
    #remove punctuation
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.remove_punctuation)
    #remove less than 3 letter words
    df['headline_text']  = df.headline_text.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))
    return df[['headline_text',  'partyaffiliation']]

  def train_preprocess(df):
    #encode labels
    df['encoded_label'] = df.apply(lambda row: Girlswhocode_PoliticalAfiiliation.encode_news_type(row['label']), axis=1)
    #convert to lower case
    df['headline_text'] = df['headline_text'].str.lower()
    #remove stop words
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.remove_stopwords)
    #spell check
    #df['headline_text'] = df['headline_text'].apply(spell_checker)
    #Lemmetize
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.lemmatize_stemming)
    #stemming
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.stemming)
    #remove punctuation
    df['headline_text'] = df['headline_text'].apply(Girlswhocode_PoliticalAfiiliation.remove_punctuation)
    #remove less than 3 letter words
    df['headline_text']  = df.headline_text.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))
    return df[['headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo', 'partyaffiliation', 'context', 'encoded_label']]

    

  def encode_party_affiliation_type(input_label):
      labels = ['democrat','republican','independent']
      if input_label not in labels:
        return str('other')
      else:
        return input_label

  def convert_partyaffiliation_category(df):
      partyaffiliation_dict = {'independent':4, 'other':3, 'democrat':2, 'republican':1}
      pa = []
      for index,row in df.iterrows():
        pa.append(partyaffiliation_dict[row['partyaffiliation']])
      return pa

  def tag_headline(df, label):
      tagged_text = []
      for index, row in df.iterrows():
        tagged_text.append(TaggedDocument(words=word_tokenize(row['headline_text']), tags=[row[label]]))
      return tagged_text

  def get_issues_vector(df_testing,issues):
      text_test =  ' '.join(issues)
      vector = []
      for row in df_testing.headline_text:
        # tokenization 
        head_line = str(row)
        text_list = word_tokenize(text_test)  
        headline_list = word_tokenize(head_line)
        #print(headline_list)
      
        # sw contains the list of stopwords 
        sw = stopwords.words('english')  
        l1 =[];l2 =[] 
        # remove stop words from the string 
        X_set = {w for w in text_list} #if not w in sw}  
        Y_set = {w for w in headline_list} #if not w in sw}
        # form a set containing keywords of both strings  
        rvector = X_set.union(Y_set)  
        for w in rvector: 
          if w in X_set: l1.append(1) # create a vector 
          else: l1.append(0) 
          if w in Y_set: l2.append(1) 
          else: l2.append(0) 
        c = 0
        #cosine formula  
        for i in range(len(rvector)): 
            c+= l1[i]*l2[i] 
        if c == 0:
          vector.append(0)
        else:
          cosine = c / float((sum(l1)*sum(l2))**0.5) 
          #print("similarity: ", cosine)
          vector.append(cosine)
      return vector

  def computeTF(df, dictionary):
    TF = []
    dict_words = dictionary['word'].unique()
    for index, row in df.iterrows():
        row_freq = []
        words = row['headline_text'].split()
        for i in range(len(dict_words)):
            frequency = float(words.count(dict_words[i])/len(dict_words))
            row_freq.append(frequency)
        TF.append(row_freq)
        #print(TF)
    return TF

  #Calculate IDF for the dictionary
  import math
  def computeIDF(df, dictionary):
    IDF = []
    dict_words = dictionary['word'].unique()
    num_of_docs = len(df)
    for i in range(len(dict_words)):
        count = 0
        for index,row in df.iterrows():
            if dict_words[i] in row['headline_text']:
                count += 1
        if count == 0:
          IDF.append(0)
        else:
          IDF.append(math.log(num_of_docs/count))
        #print(IDF)
    return IDF

  #Calculate TF-IDF for each headline text based on the dictionary created
  def computeTFIDF(TF, IDF):
    TFIDF = []
    IDF = np.asarray(IDF)
    #print(IDF)
    #print(IDF.T)
    for j in TF:
        tfidf = np.asarray(j) * IDF.T
        TFIDF.append(tfidf)
    return TFIDF

  def sentiment_analyzer_scores(df):
    analyser = SentimentIntensityAnalyzer()
    sentiment_score = []
    sentiment_labels = {0:'negative', 1:'positive', 2:'neutral'}
    for index,row in df.iterrows():
        score = analyser.polarity_scores(row['headline_text'])
        values = [score['neg'], score['pos'], score['neu']]
        max_index = values.index(max(values))
        data = {'senti_score':score, 'senti_label':sentiment_labels[max_index], 'senti_label_encode': 1+math.log(max_index+1)}
        sentiment_score.append(data)
    return sentiment_score

  #tokenization - process of splitting text to words
  def get_word_tokens(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) > 3:
            result.append(token)
    return result

  """def get_dictionary(df):
    documents = df_train[['headline_text']]
    processed_docs = documents['headline_text'].map(get_word_tokens)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)"""


  def identify_topic_number_score(text,df_train):
    print("here in LDA")
    documents = df_train[['headline_text']]
    processed_docs = documents['headline_text'].map(Girlswhocode_PoliticalAfiiliation.get_word_tokens)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    bow_vector = dictionary.doc2bow(Girlswhocode_PoliticalAfiiliation.get_word_tokens(text))
    topic_number , topic_score = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])[0]
    #print (topic_number, topic_score)
    return pd.Series([topic_number, topic_score])

  """def get_political_affiliation_test_vector(df, doc2vec_model_pa_test):
    #doc2vec_model_pa_test = Doc2Vec(documents = tagged_pa_text_test, dm=0, num_features=500, min_count=2, size=20, window=4)
    pa_test = []
    for i in range(len(df['partyaffiliation_encode'])):
        pa_value = df['partyaffiliation_encode'][i]
        pa = doc2vec_model_pa_test[pa_value] + df['sentiment_encode'][i] + df['topic_score'][i] +df['republican_vector'][i]+df['democrats_vector'][i]+df['liberterian_vector'][i]
        pa_test.append(pa)
    return pa_test"""

  def predict(model, X_test):
    y_pred = model.predict(X_test)
    predicted_proba = model.predict_proba(X_test)[:,1]
    return y_pred, predicted_proba

  def DATAMINERS_getPartyAffiliationScore(headline, party):
      test = [[headline, party]]

      #creating the dataframe
      df_testing = pd.DataFrame(test,columns=['headline_text','partyaffiliation'])
      print(df_testing.columns)

      #preprocessing of the text
      df_testing = Girlswhocode_PoliticalAfiiliation.text_preprocess(df_testing)

      #encoding partyaffiliation
      pa_encode_test = pd.DataFrame(Girlswhocode_PoliticalAfiiliation.convert_partyaffiliation_category(df_testing))
      df_testing['partyaffiliation_encode'] = pa_encode_test

      #issues related to democrats,republicans and libertarian
      democrats_issues = ['minimum wage', 'health care','education','environment','renewable energy','fossil fuels']
      libertarian_issues = ['taxes','economy','civil liberties','crime and justice','foreign policy','healthcare','gun ownership','war on drugs','immigration']
      republican_issues =['Abortion and embryonic stem cell research','Civil rights','Gun ownership','Drugs','Education','Military service','Anti-discrimination laws']

      df_testing['republican_vector'] = Girlswhocode_PoliticalAfiiliation.get_issues_vector(df_testing,republican_issues)
      
      df_testing['democrats_vector'] = Girlswhocode_PoliticalAfiiliation.get_issues_vector(df_testing,democrats_issues)
      
      df_testing['liberterian_vector'] = Girlswhocode_PoliticalAfiiliation.get_issues_vector(df_testing,libertarian_issues)
      print(df_testing.columns)

      party_affiliation_dict = pd.read_csv('/content/drive/MyDrive/MLFall2020/girlswhocode/datasets/Alternusvera_dataset/party_affiliaiton_dict.csv')

      TF_scores = Girlswhocode_PoliticalAfiiliation.computeTF(df_testing, party_affiliation_dict)
      IDF_scores = Girlswhocode_PoliticalAfiiliation.computeIDF(df_testing, party_affiliation_dict)
      TFIDF_scores = Girlswhocode_PoliticalAfiiliation.computeTFIDF(TF_scores, IDF_scores)

      #append tfidf score to the dataframe
      tfidf_test = []
      for i in range(len(TFIDF_scores)):
        tfidf_test.append(max(TFIDF_scores[i]))

      df_testing['tfidf'] = tfidf_test

      #tfidf wrt issues, get the issues dictionary which has the frequency of all the issues
      issues_dict = pd.read_csv('/content/drive/MyDrive/MLFall2020/girlswhocode/datasets/Alternusvera_dataset/new_dict.csv')

      TFissues_scores = Girlswhocode_PoliticalAfiiliation.computeTF(df_testing, issues_dict)
      IDFissues_scores = Girlswhocode_PoliticalAfiiliation.computeIDF(df_testing, issues_dict)
      TFIDFissues_scores = Girlswhocode_PoliticalAfiiliation.computeTFIDF(TFissues_scores, IDFissues_scores)

      #append issues score to the dataframe
      tfidf_issues = []
      for i in range(len(TFIDFissues_scores)):
        tfidf_issues.append(max(TFIDFissues_scores[i]))

      df_testing['issues_score'] = tfidf_issues


      #get sentiment score related to party
      sentiment_score_input = Girlswhocode_PoliticalAfiiliation.sentiment_analyzer_scores(df_testing)
      sentiment_score_input = pd.DataFrame(sentiment_score_input)
      df_testing['sentiment'] = sentiment_score_input['senti_label']
      df_testing['sentiment_encode'] = sentiment_score_input['senti_label_encode']
      print(df_testing.columns)

      #load train dataset to get the dictionary of issues to calculate the topic score of each headline text
      colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytrueocunts','pantsonfirecounts','context']
      df_train = pd.read_csv('/content/drive/MyDrive/MLFall2020/girlswhocode/datasets/Alternusvera_dataset/train.tsv', sep='\t', names = colnames,error_bad_lines=False)
      #preprocess the df_train headline_text
      df_train = Girlswhocode_PoliticalAfiiliation.train_preprocess(df_train)
      print(df_train.columns)

      


      #get topic score related to party
      df_testing[['topic_number','topic_score']] = df_testing.apply(lambda row: Girlswhocode_PoliticalAfiiliation.identify_topic_number_score(row['headline_text'],df_train), axis=1)
      print(df_testing.head())

      #get the tagged text fot the headline
      tagged_pa_text_test = Girlswhocode_PoliticalAfiiliation.tag_headline(df_testing, 'partyaffiliation_encode')
      print("tagged text",tagged_pa_text_test)
      #get the Doc2Vec for the headline_text
      print("before")
      #doc2vec_model_pa_test = Doc2Vec(documents = tagged_pa_text_test, dm=0, num_features=500, min_count=2, size=20, window=4)
      print("after")
      #get the test vector 
      #X_test = Girlswhocode_PoliticalAfiiliation.get_political_affiliation_test_vector(df_testing, doc2vec_model_pa_test)
      #def get_political_affiliation_test_vector(df, doc2vec_model_pa_test):
      #doc2vec_model_pa_test = Doc2Vec(documents = tagged_pa_text_test, dm=0, num_features=500, min_count=2, size=20, window=4)
      X_test = []
      for i in range(len(df_testing['partyaffiliation_encode'])):
          pa_value = df_testing['partyaffiliation_encode'][i]
          pa = df_testing['sentiment_encode'].iloc[i] + df_testing['topic_score'].iloc[i] +df_testing['republican_vector'].iloc[i]+df_testing['democrats_vector'].iloc[i]+df_testing['liberterian_vector'].iloc[i]+df_testing['tfidf'].iloc[i]+df_testing['issues_score'].iloc[i]
          X_test.append(pa)
        
      X_test = np.array(X_test)
      X_test = X_test.reshape(-1,1)
      #load the saved model
      fakenews_classifier = pickle.load(open('/content/drive/MyDrive/MLFall2020/girlswhocode/models/political_affiliation_model.sav', 'rb'))

      #predict the test input
      binary_value_predicted, predicted_proba = Girlswhocode_PoliticalAfiiliation.predict(fakenews_classifier, X_test)

      return (1 - float(predicted_proba))