"""
Standard imports:
"""
import pickle
import pandas as pd
from string import punctuation
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
import string
from nltk.stem import WordNetLemmatizer, SnowballStemmer
stemmer = SnowballStemmer('english')


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
