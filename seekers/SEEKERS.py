"""
Standard imports:
"""
import pickle
import pandas as pd
from string import punctuation
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')

import re
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer = nltk.WordNetLemmatizer()
from string import punctuation
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
from transformers import BertTokenizer
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler 
from scipy import sparse
from transformers import BertForSequenceClassification
import torch.nn.functional as F


"""Loading the pickled model and predicting the score"""
sw = stopwords.words('english')
class Seekers_StanceDetection():

  
  stance_score = {'agree': 0.0,'disagree': 0.8,'discuss':0.3,'unrelated': 1.0 }
  
  def __init__(self, filename):
    self.model = self.load(filename)
  
  def remove_stop_and_short_words(self,text):
    text = [word.lower() for word in text.split() if (word.lower() not in sw) and (len(word)>3)]
    return " ".join(text)
  
  def lemmatize_stemming(self,text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
  
  def remove_punctuation(self,text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)
  
  def process_data(self,text):
    # print ('Input Text :: ' + text)
    text = self.remove_stop_and_short_words(text)
    #print('Stopwords and short words removed :: ' + text)
    text = self.lemmatize_stemming(text)
    #print('Lemmatized :: ' + text)
    text = self.remove_punctuation(text)
    #print('Punctuation removed :: ' + text)
    return text

  def load(self, path):
    with open(path, 'rb') as file:
      return pickle.load(file)

  def predict(self, text):
    processedText = self.process_data(text)
    result = self.model.predict_proba([processedText])
    predictedStance = self.model.predict([processedText])[0]
    if predictedStance == 2:
        result=0.49
    elif predictedStance == 1:
        result=0.11
    else: 
        result=0.8
    return result

#print("op ", Seekers_StanceDetection("randomforest.sav").predict("text here"))




class Seekers_ClickBait():

  def __init__(self, multinomialFile, tfidfFile):
    self.multinomialModel = self.load(multinomialFile)
    self.tfidfModel = self.load(tfidfFile)

  def load(self, path):
    with open(path, 'rb') as file:
      return pickle.load(file)

  def tokenization(self,text):
    lst=text.split()
    return lst

  def lowercasing(self,lst):
      new_lst=[]
      for i in lst:
          i=i.lower()
          new_lst.append(i)
      return new_lst
 
  def remove_punctuations(self,lst):
      new_lst=[]
      for i in lst:
          for j in punctuation:
              i=i.replace(j,'')
          new_lst.append(i)
      return new_lst


  def remove_numbers(self,lst):
      nodig_lst=[]
      new_lst=[]
      for i in lst:
          for j in self.digits:    
              i=i.replace(j,'')
          nodig_lst.append(i)
      for i in nodig_lst:
          if i!='':
              new_lst.append(i)
      return new_lst

  def remove_stopwords(self,lst):
      stop=stopwords.words('english')
      new_lst=[]
      for i in lst:
          if i not in stop:
              new_lst.append(i)
      return new_lst
  
  def remove_spaces(self,lst):
    new_lst=[]
    for i in lst:
        i=i.strip()
        new_lst.append(i)
    return new_lst

  
  def lemmatzation(self, lst):
    lemmatizer=nltk.stem.WordNetLemmatizer()
    new_lst=[]
    for i in lst:
        i=lemmatizer.lemmatize(i)
        new_lst.append(i)
    return new_lst

  def predict(self,text):
    dfrme = pd.DataFrame(index=[0], columns=['text'])
    dfrme['text'] = text
    predict=dfrme['text'].apply(self.tokenization)
    predict=predict.apply(self.lowercasing)
    predict=predict.apply(self.remove_punctuations)
    predict=predict.apply(self.remove_stopwords)
    predict=predict.apply(self.remove_spaces)
    predict=predict.apply(self.lemmatzation)
    predict =predict.apply(lambda x: ''.join(i+' ' for i in x))
    text = self.tfidfModel.transform(predict)
    train_arr=text.toarray()
    probValue = self.multinomialModel.predict_proba(train_arr)[:,1][0]
    return probValue

#print('op ', Seekers_ClickBait('Seekers_ClickBait.sav','tfidf.sav').predict('text here'))


class Seekers_Spam():

  def __init__(self, loaded_tdIdfModel, loaded_model):
    self.loaded_tdIdfModel = self.load(loaded_tdIdfModel)
    self.loaded_model = self.load(loaded_model)
    
  def load(self, path):
    with open(path, 'rb') as file:
      return pickle.load(file)
    
  def tokenization(self,text):
    lst=text.split()
    return lst

  def predict(self, text):
    dfrme = pd.DataFrame(index=[0], columns=['text'])
    dfrme['text'] = text
    predict=dfrme['text'].apply(self.tokenization)
    predict = predict.apply(lambda x: ''.join(i+' ' for i in x))
    text = self.loaded_tdIdfModel.transform(predict)
    result = self.loaded_model.predict_proba(text.toarray())[:,1][0]
    return result

#print('op ', Seekers_Spam('TFidfvectorizer.sav','final_SpamModel.sav').predict('text here'))



# -*- coding: utf-8 -*-
"""Seekers_BertMcc.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1i4iQgvzkWh8TUDh9KmYPdgmIiXK85rRE
"""


class Seekers_BertMcc():

    def __init__(self, pathModelBR): 
        self.modelBR = BertForSequenceClassification.from_pretrained(pathModelBR)
        # model = BertForSequenceClassification.from_pretrained("/content/")


    def cleaning(self, raw_data):
    
      # 1. Remove non-letters/Special Characters and Punctuations
      data = re.sub("[^a-zA-Z]", " ", str(raw_data))
      
      # 2. Convert to lower case.
      data =  data.lower()
      
      # 3. Tokenize.
      data_words = nltk.word_tokenize(data)
      
      # 4. Convert the stopwords list to "set" data type.
      stops = set(nltk.corpus.stopwords.words("english"))
      
      # 5. Remove stop words. 
      words = [w for w in  data_words  if not w in stops]
      
      # 6. Lemmentize 
      wordnet_lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]
      
      # 7. Stemming
      stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in wordnet_lem ]
      
      # 8. Join the stemmed words back into one string separated by space, and return the result.
      return " ".join(wordnet_lem)

    def scaleTruth(self, mostly_true_count,half_true_count,barely_true_count,false_count,pants_on_fire_count):  

      max_len = 300
      cols = ['mostly_true_count','half_true_count','barely_true_count','false_count','pants_on_fire_count']
      history_features = pd.DataFrame(np.array([[mostly_true_count,half_true_count,barely_true_count,false_count,pants_on_fire_count]]),columns=cols)
                             
      scaler = StandardScaler()
      scaler.fit(history_features)
      StandardScaler(copy=True, with_mean=True, with_std=True)
      history_features = scaler.transform(history_features)
      num_rows, num_cols = history_features.shape
      suffix = np.zeros([num_rows, max_len-num_cols])
      history_features = np.concatenate((history_features,suffix),axis=1)
      return history_features

    def tokenize_text(self, text,mostly_true_count,half_true_count,barely_true_count,false_count,pants_on_fire_count):
      textline = []
      # Load the BERT tokenizer.
      X_t = self.scaleTruth(mostly_true_count,half_true_count,barely_true_count,false_count,pants_on_fire_count)
      tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
      input_ids = []
      attention_masks = []
      textline.append(text)
      for sent in textline:

          encoded_dict = tokenizer.encode_plus(
                              sent,                      # Sentence to encode.
                              add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                              max_length = 300,      # Pad & truncate all sentences.
                              pad_to_max_length = True,
                              return_attention_mask = True,   # Construct attn. masks.
                              return_tensors = 'pt',     # Return pytorch tensors.
                      )
          
          # Add the encoded sentence to the list.    
          input_ids.append(encoded_dict['input_ids'])
          
          
          # And its attention mask (simply differentiates padding from non-padding).
          attention_masks.append(encoded_dict['attention_mask'])

      # Convert the lists into tensors.
      input_ids = torch.cat(input_ids, dim=0)
      attention_masks = torch.cat(attention_masks, dim=0)

      input_ids = input_ids.numpy().astype(float)
      for j in range(0,299):
        if(input_ids[0][j]==102):
          input_ids[0][j] = X_t[0][0]
          input_ids[0][j+1] = X_t[0][1]
          input_ids[0][j+2] = X_t[0][2]
          input_ids[0][j+3] = X_t[0][3]
          input_ids[0][j+4] = X_t[0][4]
          input_ids[0][j+5] = 102
          break
              
      input_ids = torch.from_numpy(input_ids) 

      (test_input_ids,  
      test_attention_masks) = (input_ids, attention_masks)
      return test_input_ids, test_attention_masks

  
    def get_bert_predictions(self, textData, source, context, mostly_true_count, half_true_count, barely_true_count, false_count, pants_on_fire_count):

      #Adding some latent variables from the data
      text     =  textData+ source + context
      text     = self.cleaning(text)
      b_input_ids, b_input_mask = self.tokenize_text(text,mostly_true_count,half_true_count,barely_true_count,false_count,pants_on_fire_count)
      
      #storing values to GPU
      b_input_ids.cuda()
      b_input_mask.cuda()

      model = self.modelBR #BertForSequenceClassification.from_pretrained("/content/")

      # model.cuda()
      desc = model.cuda()
      
      # Put model in evaluation mode
      model.eval()
      
      # Tracking variables 
      predictions = []
      
      with torch.no_grad():
      # Forward pass, calculate logit predictions
          b_input_ids = torch.tensor(b_input_ids).cuda().long()
          b_input_mask = torch.tensor(b_input_mask).cuda().long()
          outputs = model(b_input_ids, token_type_ids=None, 
                              attention_mask=b_input_mask)
      
      logits = outputs[0]
      
      # Move logits to CPU
      logits = logits.detach().cpu().numpy()
   
      # Store predictions  
      predictions.append(logits)
      
      # Combine the results across the batches.
      predictions = np.concatenate(predictions, axis=0)
      tensorProbability = F.softmax(torch.tensor(predictions))
      # Take the highest scoring output as the predicted label.
      predicted_labels = np.argmax(predictions, axis=1)
      tensorProbability = tensorProbability.numpy()
      predicted = str(predicted_labels[0])
      arr_fake = np.array([tensorProbability[0][3],tensorProbability[0][4],tensorProbability[0][5]])
      fakeness_probability = np.max(arr_fake)
      return int(predicted), fakeness_probability