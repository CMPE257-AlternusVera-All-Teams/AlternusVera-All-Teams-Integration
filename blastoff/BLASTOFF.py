import pandas as pd
import numpy as np
import csv
import gensim
import re
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from scipy import sparse
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize 
nltk.download('punkt', quiet=True)
from zipfile import ZipFile
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from googlesearch import search
import keras


#Imports
import torch
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from google_drive_downloader import GoogleDriveDownloader as gdd


class Context_Veracity():
  def __init__(self, veracity_models):

    self.model = None 
    colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytruecounts','pantsonfirecounts','context', 'text']

    # unpickling models
    names = ["Random Forest"]
    with ZipFile(veracity_models, 'r') as myzip:
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





class SensaScorer():
  def __init__(self):
    # self.sensa_dict = {1:'Barely sensationalist',0: 'Not sensationalist',2:'Sensationalist'}
    self.sensa_dict = {1:0.55,0:0.25,2:0.95}
    self.label_dict = {'Barely sensationalist': 1, 'Not sensationalist': 0, 'Sensationalist': 2}
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                              num_labels=len(self.label_dict),
                                                              output_attentions=False,
                                                              output_hidden_states=False)
    self.model.to(self.device)
    gdd.download_file_from_google_drive(file_id='1SpfmiCq2a2aXTXvFW6cHnm-0eBCpcyxY',
                                  dest_path='./sensationalism_BERT_best.model',
                                  unzip=False)
    
  def getScore(self,title):
    self.model.load_state_dict(torch.load('sensationalism_BERT_best.model', 
                                      map_location=torch.device('cpu')))
    prediction = self.evaluateSentimentScore(title)
    
    return prediction

  #1 Function that receives a String with max 250 characters
  def evaluateSentimentScore(self,news_title):
    #Add title string to dataframe
    df = pd.DataFrame(columns=['English','label'])
    df.loc[0]=[news_title,0]
    #Instantiate BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', 
                                              do_lower_case=True)
    #Econded record(s)
    encoded_data = tokenizer.batch_encode_plus(
      df.English.values, 
      add_special_tokens=True, 
      return_attention_mask=True, 
      pad_to_max_length=True, 
      max_length=256, 
      return_tensors='pt'
      )
    #Get inputs_ids_val and attention_masks_val
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor([0])
    #Tensor Dataset
    dataset = TensorDataset(input_ids, attention_masks, labels)
    #Create dataloader
    dataloader = DataLoader(dataset,
                            sampler=SequentialSampler(dataset),
                            batch_size=1)
    #Call evalation method and return values
    return self.evaluate(dataloader)

  # 2 Define Evaluation method Similar to evaluate but 
  def evaluate(self,dataloader):
    self.model.eval()

    loss_val_total = 0
    predictions, true_vals = [], []
    
    for batch in dataloader:
        
        batch = tuple(b.to(self.device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  'labels':         batch[2],
                  }

        with torch.no_grad():        
            outputs = self.model(**inputs)
            
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    preds = np.concatenate(predictions, axis=0)
    preds_flat = np.argmax(preds, axis=1).flatten()
    return self.sensa_dict[preds_flat[0]]
