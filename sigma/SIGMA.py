import pandas as pd
import numpy as np
import pickle 


class NetworkBasedPredictor():

  def __init__(self, filename):
    self.model = self.load(filename)

  def __convert2vector(self, tweetToPredict, nlp): 
        textToPredict = str(tweetToPredict)
        review = nlp(textToPredict)
        nlpx_tweet = []
        vector_tweet=0
        for token in review:
          if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 2 and token.text != 'https' ):
                  vector_tweet += (nlp.vocab[token.text].vector)
        if len(review) != 0:
            vector_tweet = vector_tweet / len(review)
        nlpx_tweet.append(vector_tweet)
        df_test_text = pd.DataFrame(nlpx_tweet) 
        return df_test_text
  
  def __convert_prediction(self, prediction):
    r = [0.16, 0.33, 0.49, 0.66, 0.83, 0.96]
    return r[prediction[0]-1]

  def load(self, path):
    with open(path, 'rb') as file:  
       return pickle.load(file)

  def predict(self, text, nlp, source=0):
    df = self.__convert2vector(text, nlp)
    df['node_rank'] = 0
    return self.__convert_prediction(self.model.predict(df))


true_venue_labels = ['news','interview','television','show', 'speech', 'reporters', 'debate', 'newsletter', 'press', 'CNN', 'ABC', 'CBS', 'video', 'conference', 'official', 'book']
false_venue_labels = ['website', 'tweet', 'mail', 'e-mail', 'mailer', 'web', 'site', 'meme', 'comic', 'advertisement', 'ad', 'blog', 'flier', 
                'letter', 'social', 'tweets', 'internet', 'message', 'campaign', 'post', 'facebook', 'handout', 'leaflet', 'letter', 'fox' ]

true_statement_labels = ['original','true','mostly-true','half-true']
false_statement_labels = ['barely-true','false','pants-fire']


class VerifiableAuthenticity():

  #Intialising and loading the pickled model
  def __init__(self, filename):
    # Load Model
    self.model = None
    with open(filename, 'rb') as file: 
      self.model = pickle.load(file)
     # print(Pickled_Model)

  # Function to calculate the venue score
  def simplify_venue_label(self, venue_label):
    if venue_label is np.nan:
      return 0;
    words = venue_label.split(" ")
    for s in words:
      if s in true_venue_labels:
        return 1
      elif s in false_venue_labels:
        return 0
    else:
        return 1


  def getAuthenticityScoreByVenue(self, src):
    x = self.simplify_venue_label(src)
    xTrain = np.array(x).reshape(-1, 1)
    xPredicted = self.model.predict(xTrain)
    xPredicedProb = self.model.predict_proba(xTrain)[:,1]
    return float(xPredicedProb), xPredicted


  def predict(self, statement='', venue=''):
    concatStatement = ''
    for str1 in statement:
      concatStatement += str1+ ' '
    venueAuth, _ = self.getAuthenticityScoreByVenue(venue)
    #print(" values ", venueAuth, probValue)
    score = 0.7 * venueAuth + 0.3 * 0.5
    #print("score =  ", score)
    return float(score)

class Credibility():

    def __init__(self, filename):
        self.model = self.load(filename)
   
    def predict( self, text, nlp ):
        vector = self.__convert2vector(text, nlp)
        predictTestCD = self.model.predict(vector)
        predictTestCD = int(predictTestCD[0])
        if predictTestCD == 0: 
          resultsCD = 'Non-Credible'
          factorCD = 0.2
        elif predictTestCD == 1 :
          resultsCD = 'Credible'
          factorCD = 0.8
        return resultsCD, factorCD


    def load( self, model2load ):
        with open(model2load, 'rb') as file:  
            Pickled_Model = pickle.load(file)

        # msg = "load a model." + model2load
        return Pickled_Model


    def save(self, filename):
        with open(filename, 'wb') as file:  
            pickle.dump(self.model, file)
        msg = "saved model " + filename 
        return msg

    def __convert2vector(self, tweetToPredict, nlp): 
        textToPredict = str(tweetToPredict)
        review = nlp(textToPredict)
        nlpx_tweet = []
        vector_tweet=0
        n_tokens=0
        for token in review:
          if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 2 and token.text != 'https' ):
                  vector_tweet += (nlp.vocab[token.text].vector)
                  n_tokens+=1 
        if n_tokens != 0:
            vector_tweet = vector_tweet / n_tokens
        nlpx_tweet.append(vector_tweet)
        df_test_text = pd.DataFrame(nlpx_tweet) 
        return df_test_text 


class MalicousAccount():

    def __init__( self, filename ):
        self.model = self.load(filename)

    def predict( self, tweetText, nlp ): 
        tweetVector = self.__convert2vector(tweetText, nlp)
        predictionResult =  self.model.predict(tweetVector)
        botScoreResult, labelResult = self.__predictionScore(predictionResult) 
        # print('label: ', predictionResult)
        # print('label: ', labelResult)
        # print('score: ', botScoreResult)
        return predictionResult[0], botScoreResult, labelResult
   
   
    def load( self, model2load ):
        with open(model2load, 'rb') as file:  
            Pickled_Model = pickle.load(file)
        return Pickled_Model


    def save(self, filename):
        with open(filename, 'wb') as file:  
            pickle.dump(self.model, file)
        msg = "saved model " + filename 
        return msg
    
    # Convert to verctor using word2Vec
    def __convert2vector(self, tweetToPredict, nlp): 
        textToPredict = str(tweetToPredict)
        review = nlp(textToPredict)
        nlpx_tweet = []
        vector_tweet=0
        n_tokens=0
        for token in review:
          if(token.pos_ == 'NOUN' or token.pos_ == 'VERB' or token.pos_ == 'ADJ' or token.pos_ == 'PROPN'):
              if (len(token.text) > 2 and token.text != 'https' ):
                  # print(token.text, " ---> ", token.pos_)
                  vector_tweet += (nlp.vocab[token.text].vector)
                  # print(vector_tweet) 
                  n_tokens += 1 
        if n_tokens != 0:
            vector_tweet = vector_tweet / n_tokens
        nlpx_tweet.append(vector_tweet)
        df_test_text = pd.DataFrame(nlpx_tweet) 
        return df_test_text


    def __predictionScore(self, prediction):
        # n_classes = ['Human', 'cyborg', 'botWallE', 'botT800' ]
        if prediction == 1: 
          label = 'Human'
          botScore = 0.90 #(1-0.8)/2
        elif prediction == 2:
          label = 'cyborg'
          botScore = 0.70 #(0.8-0.6)/2
        elif prediction == 3: 
          label = 'botWallE'
          botScore = 0.50 #(0.6-0.4)/2
        elif prediction == 4: 
          label = 'botT800'      
          botScore = 0.30 #(0.4-0.2)/2
        else: 
          label = 'Allient'      
          botScore = 0.10 #(0.2-0.0)/2      
        return botScore, label







     
