import transformers
from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import autogluon.core as ag
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPrediction as task2
from autogluon.text import TextPrediction as task
import pickle 

class CerealKillers_SentimentClassifier(nn.Module):
  def __init__(self, filename):
    super(CerealKillers_SentimentClassifier, self).__init__()
    self.tokenizer = BertTokenizer.from_pretrained(filename)
    self.bert = BertModel.from_pretrained(filename)
    self.class_names = ['true','pants-fire']
    self.n_classes = len(self.class_names)
    self.drop = nn.Dropout(p=0.3)
    self.out = nn.Linear(self.bert.config.hidden_size, self.n_classes)

  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

  def predict(self, review_text):
    encoded_review = self.tokenizer.encode_plus(
      review_text,
      max_length=512,
      add_special_tokens=True,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )
    input_ids = encoded_review['input_ids']
    attention_mask = encoded_review['attention_mask']
    output = self.forward(input_ids, attention_mask)
    _, prediction = torch.max(output,1)
    bert_score = 0
    if prediction.item() == 0:
      bert_score = 1
    elif prediction.item() == 1:
      bert_score = 2
    elif prediction.item() == 2:
      bert_score = 3
    elif prediction.item() == 3:
      bert_score = 4
    elif prediction.item() == 4:
      bert_score = 5
    else:
      bert_score = 6
    return bert_score


class CerealKillers_PsychologicalUtility:
  def __init__(self, rank_file, sts_file):
    self.predictor_rank = task2.load(rank_file)
    self.predictor_sts  = task.load(sts_file)

  def predict(self, text, bt=0,f=0,ht=0,mt=0,po=0):
    rank_test  = pd.DataFrame(np.array([[bt,f,ht,mt,po]]), columns=['BARELY TRUE', 'FALSE', 'HALF TRUE', 'MOSTLY TRUE', 'PANTS ON'])
    rank_score = self.predictor_rank.predict(rank_test)
    sen_score  = self.predictor_sts.predict_proba({"text": [text]})

    #print("RANK:", rank_score[0]/6)
    #print("SENTI:",sen_score[0][1])
    total_score = 0.5*rank_score[0]/6 + 0.5*sen_score[0][1]
    return total_score


class CerealKillers_SocialCredibility:

  def __init__(self, filename):
    self.model = self.load(filename)


  def load(self, path):
    with open(path, 'rb') as file:  
       return pickle.load(file)

  def predict(self, data):
    X = pd.DataFrame([data], columns=['followers','favorites','friends','listed_count','statuses_count','status_retweeted','status_favorited'])
    #RFC = pickle.load( open("/content/CerealKillers_AlternusVera/SC/SocialCredibility.pkl", "rb") )
    y = self.model.predict(X)
    return y[0]
