import pandas as pd
import numpy as np
import csv
import gensim
from sklearn.model_selection import train_test_split
import re
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from scipy import sparse
from google_drive_downloader import GoogleDriveDownloader as gdd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
nltk.download('punkt')
from zipfile import ZipFile
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from googlesearch import search
import keras

class Context_Veracity():
  def __init__(self):
    gdd.download_file_from_google_drive(file_id='18NyBuNHikHiUzrAC4oOMUOO5KAaKouEY',
                                  dest_path='./context_veracity_models.zip',
                                  unzip=False)
    self.model = None 
    colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytruecounts','pantsonfirecounts','context', 'text']

    # unpickling models
    names = ["Random Forest"]
    with ZipFile('context_veracity_models.zip', 'r') as myzip:
        for name in names:
            self.model = pickle.load(myzip.open(f'{name}_model.pickle'))
            #print(clf_reload)


  def get_veracity_scores(self, title):
    #calculate title_count on veracity
    source_count = self.find_similar_articles(title)
    if(source_count > 3):
      veracity = 1
    else:
      veracity = 0
    return self.get_veracity(veracity, source_count)
  
  def get_source_count_and_veracity(self, title):
  #calculate title_count on veracity
    source_count = self.find_similar_articles(title)
    if(source_count > 3):
      veracity = 1
    else:
      veracity = 0
    return (source_count,veracity)

  def get_veracity(self, veracity, title_count):
    df = pd.DataFrame(columns=['veracity', 'title_count'])
    df.loc[0]=[veracity, title_count]
    result = self.model.predict(df)
    return result

  def remove_unnecessary_noise(self, text_messages):
    text_messages = re.sub(r'\\([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])\\([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])\\([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])', ' ', text_messages)
    text_messages = re.sub(r'\\([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])\\([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])([a-z]|[A-Z]|[0-9])', ' ', text_messages)
    text_messages = re.sub(r'\[[0-9]+\]|\[[a-z]+\]|\[[A-Z]+\]|\\\\|\\r|\\t|\\n|\\', ' ', text_messages)

    return text_messages

  def preproccess_text(self, text_messages):
    # change words to lower case - Hello, HELLO, hello are all the same word
    processed = text_messages.lower()

    # Remove remove unnecessary noise
    processed = re.sub(r'\[[0-9]+\]|\[[a-z]+\]|\[[A-Z]+\]|\\\\|\\r|\\t|\\n|\\', ' ', processed)

    # Remove punctuation
    processed = re.sub(r'[.,\/#!%\^&\*;\[\]:|+{}=\-\'"_”“`~(’)?]', ' ', processed)

    # Replace whitespace between terms with a single space
    processed = re.sub(r'\s+', ' ', processed)

    # Remove leading and trailing whitespace
    processed = re.sub(r'^\s+|\s+?$', '', processed)
    return processed
  
  def news_title_tokenization(self, message):
    stopwords = nltk.corpus.stopwords.words('english')
    tokenized_news_title = []
    ps = PorterStemmer()
    for word in word_tokenize(message):
        if word not in stopwords:
            tokenized_news_title.append(ps.stem(word))

    return tokenized_news_title

  def find_similar_articles(self, news):
    
    news_title_tokenized = ''
    
    if(re.match(r'^https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)$', news)):
        news_article = Article(news)
        news_article.download()
        news_article.parse()
        news_title_tokenized = self.news_title_tokenization(self.preproccess_text(news_article.title))
    else:
        news_title_tokenized = self.news_title_tokenization(self.preproccess_text(news))

    search_title = ''
    for word in news_title_tokenized:
      search_title = search_title + word + ' '

    #print(search_title)
    count = 0
    post = 0
    post_true = False
    non_credit_sources = ['facebook', 'twitter', 'youtube', 'tiktok']
    for j in search(search_title, num=1, stop=10, pause=.30): 
      #print(j)
      post_true = False
      for k in non_credit_sources:
        if k in j:
          post+= 1
          post_true = True
      if(post_true == False):
        count+= 1
    #print("Count is", count, "and Post is", post)  
    
    return count
  
  def encode(self, X_train):
    bcv_tc = []
    bcv_v = []
    for s in X_train['Statement'].tolist():
        tc, v = self.get_source_count_and_veracity(s)
        bcv_tc.append(tc)
        bcv_v.append(v)
    bcv_d = {'title_count': bcv_tc, 'veracity': bcv_v}
    bcv_e_X_train = pd.DataFrame(data=bcv_d)
    
    gdd.download_file_from_google_drive(file_id='1Pu0D6GffO5fBgXVCVnKcEAPr9lrbAYfK',
                                      dest_path='./bcv_encoder.zip',
                                      unzip=False)
    archive = ZipFile('bcv_encoder.zip')
    for file in archive.namelist():
        archive.extract(file, '/content/')
    bcv_encoder = keras.models.load_model('/content/bcv_encoder')
    bcv_e_X_train = bcv_encoder.predict(bcv_e_X_train[['title_count', 'veracity']])
    
    return bcv_e_X_train
  
  #Method for Liar dataset
  def liar_encode(self, X_train):
    train_news = X_train
    #train_news = train_news.dropna(subset=['label'])
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder

    # creating instance of labelencoder
    labelencoder = LabelEncoder()
    # Assigning numerical values and storing in another column
    #train_news['label_cat'] = labelencoder.fit_transform(train_news['label'])
    train_news['veracity'] = 0
    #Find veracity
    for index, row in train_news.iterrows():
      if (train_news.loc[index, 'barelytruecounts'] > 4) | (train_news.loc[index, 'falsecounts'] >= 2) | (train_news.loc[index, 'pantsonfirecounts'] >= 1):
        train_news.loc[index,'veracity'] = 0
      else:
        if (train_news.loc[index, 'halftruecounts'] >= 2) | (train_news.loc[index, 'mostlytruecounts'] >= 1):
          train_news.loc[index,'veracity'] = 1
        
    train_news = train_news.dropna(how='any',axis=0)
    train_news = train_news.rename(columns={'headline_text': 'Statement', 'speaker': 'Source'})

    #Find source count
    col_to_avg = ['barelytruecounts', 'falsecounts', 'pantsonfirecounts', 'halftruecounts', 'mostlytruecounts']
    train_news['title_count'] = train_news[col_to_avg].mean(axis=1)
    train_news['title_count'] = train_news['title_count'].astype(int)
    
    gdd.download_file_from_google_drive(file_id='1Pu0D6GffO5fBgXVCVnKcEAPr9lrbAYfK',
                                      dest_path='./bcv_encoder.zip',
                                      unzip=False)
    archive = ZipFile('bcv_encoder.zip')
    for file in archive.namelist():
        archive.extract(file, '/content/')
    bcv_encoder = keras.models.load_model('/content/bcv_encoder')
    bcv_e_X_train = bcv_encoder.predict(train_news[['title_count', 'veracity']])
    
    return bcv_e_X_train