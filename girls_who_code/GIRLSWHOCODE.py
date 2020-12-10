# -*- coding: utf-8 -*-
"""AlternusVera_Topic_LDA_Sprint4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gxwoiBlbXvC4Qehnx07hlIG_ju37H_dh
"""

import pickle
import pandas as pd

class Topics_with_LDA_Bigram: 

  def __init__(self, filenameModelLog): 
      self.modelLog = self.__load(filenameModelLog)

  def __load(self, path):
      with open(path, 'rb') as file:
          return pickle.load(file) 

  def encodeLabel(self, df):
    df.Label[df.Label == 'FALSE'] = 0
    df.Label[df.Label == 'half-true'] = 1
    df.Label[df.Label == 'mostly-true'] = 1
    df.Label[df.Label == 'TRUE'] = 1
    df.Label[df.Label == 'barely-true'] = 0
    df.Label[df.Label == 'pants-fire'] = 0
    return df

  def sent_to_words(self, sentences):
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

  def data_preprocess(self, df):
    import re
    dt_processed = df.News.values.tolist()
    # Remove new line characters
    dt_processed = [re.sub(r'[^\w\s]','',sent) for sent in dt_processed]
    dt_processed = [re.sub("'"," ",sent) for sent in dt_processed]
    data_words_processed = list(self.sent_to_words(dt_processed))
    return data_words_processed
  
    # Define functions for stopwords, bigrams, trigrams and lemmatization
  def remove_stopwords(self, texts):
    from nltk.corpus import stopwords
    from gensim.utils import simple_preprocess
    import nltk
    nltk.download('stopwords')
    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

  def lemmatization(self, texts):
    # spacy for lemmatization
    import spacy
    """https://spacy.io/api/annotation"""
    nlp = spacy.load('en', disable=['parser', 'ner'])
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc])
    return texts_out

  def extract_bigrams(self, data):
    from nltk.util import ngrams
    n_grams = ngrams(data, 2)
    return ['_'.join(grams) for grams in n_grams]    

  def create_bigrams(self):
    Bigrams = []
    for i in range(len(data_words_processed)):
      Bigrams.append(self.extract_bigrams(data_words_processed[i]))
    return Bigrams    

  def lda_model_final(self, corpus, id2word):
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel

    lda_model_bigram_scrapped = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=20, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
    return lda_model_bigram_scrapped  

  def format_topics_sentences(self, ldamodel, corpus, texts):
    import pandas as pd
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return (sent_topics_df)

  #Calculate sentiment polarity and find the max value. Normalize the encoded label values.
  def sentiment_analyzer_scores(self, df):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    import math
    analyser = SentimentIntensityAnalyzer()
    sentiment_score = []
    sentiment_labels = {0:'negative', 1:'positive', 2:'neutral'}
    for index,row in df.iterrows():
        score = analyser.polarity_scores(row['News'])
        values = [score['neg'], score['pos'], score['neu']]
        max_index = values.index(max(values))
        data = {'sentiment_score':score, 'sentiment_label':sentiment_labels[max_index], 'sentiment_label_encode': 1+math.log(max_index+1)}
        sentiment_score.append(data)
    return sentiment_score 

  #Scalar Vector for testing dataset
  def testing_dataset_vector(self, df_testing):
    vec_testing = []
    for i in range(len(df_testing['sentiment_encode'])):
        vec = df_testing['sentiment_encode'].iloc[i] + df_testing['Topic_Score'].iloc[i]
        vec_testing.append(vec)
    return vec_testing      

  def getTopicScoreBigramLDAModel(self, headline):
    import pandas as pd
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    from sklearn.linear_model import  LogisticRegression
    #load the model
    import pickle 
    import numpy as np

    cols = [[headline]]
    df_testing = pd.DataFrame(cols,columns=['News'])
    df_testing.head()

    #encoding the label from text to numeric value
    #df_testing = self.encodeLabel(df_testing)

    #data pre-process
    dt_words_processed =  self.data_preprocess(df_testing)

    # Remove Stop Words
    dt_nostops = self.remove_stopwords(dt_words_processed)

    # Form Bigrams
    dt_words_bigrams = []
    for i in range(len(dt_nostops)):
      dt_words_bigrams.append(self.extract_bigrams(dt_nostops[i]))  
        
    # Do lemmatization keeping only noun, adj, vb, adv
    dt_lemmatized = self.lemmatization(dt_words_bigrams)
    print(dt_lemmatized) 


    # Create Dictionary
    id2word = corpora.Dictionary(dt_lemmatized)

    # Create Corpus
    texts = dt_lemmatized

    # Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in texts]

      
    #Bigram LDA Model
    lda_model_bigram =  self.lda_model_final(corpus, id2word)

    #
    dt_topic_sents_keywords = self.format_topics_sentences(ldamodel=lda_model_bigram, corpus=corpus, texts=dt_words_processed)

    # Format
    dt_dominant_topic = dt_topic_sents_keywords.reset_index()
    dt_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Score', 'Keywords', 'Text']

    # Show
    dt_dominant_topic.head()

    #Distillation - Sentiment analysis score
    sentiment_score_dt = self.sentiment_analyzer_scores(df_testing)
    print(sentiment_score_dt)

    #append dataset with sentiment label and normalized encoded value
    sentiment_score = pd.DataFrame(sentiment_score_dt)
    df_testing['sentiment'] = sentiment_score['sentiment_label']
    df_testing['sentiment_encode'] = sentiment_score['sentiment_label_encode']
    df_testing['Keywords'] = dt_dominant_topic['Keywords']
    df_testing['Dominant_Topic'] = dt_dominant_topic['Dominant_Topic']
    df_testing['Topic_Score'] = dt_dominant_topic['Topic_Score']
    print(df_testing.head())

    #Get test data for classification

    X_test = np.array(self.testing_dataset_vector(df_testing))
    X_test =  X_test.reshape(-1, 1)

    predicted = self.modelLog.predict(X_test)
    predicedProb = self.modelLog.predict_proba(X_test)[:,1]


    score = 1 - float(predicedProb)
    return score
   

 #BIAS
 # -*- coding: utf-8 -*-
"""girlswhocode_Bias.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1JCKztviIQ2xT_h23kfLw1eloKMSmkFSQ
"""

class Gwc_Bias():

    def __init__(self, filenameModelBias): 
        self.modelBias = self.__load(filenameModelBias)

    def __load(self, path):
        with open(path, 'rb') as file:
            return pickle.load(file) 

    def cleaning(self, raw_news):
      import nltk
      import re
      from nltk.stem.wordnet import WordNetLemmatizer
      lemmatizer = nltk.WordNetLemmatizer()
      from nltk.corpus import stopwords
      from string import punctuation
      nltk.download('punkt')
      nltk.download('averaged_perceptron_tagger')
      nltk.download('wordnet')
      nltk.download('stopwords')

    
      # 1. Remove non-letters/Special Characters and Punctuations
      news = re.sub("[^a-zA-Z]", " ", raw_news)
      
      # 2. Convert to lower case.
      news =  news.lower()
      
      # 3. Tokenize.
      news_words = nltk.word_tokenize(news)
      
      # 4. Convert the stopwords list to "set" data type.
      stops = set(nltk.corpus.stopwords.words("english"))
      
      # 5. Remove stop words. 
      words = [w for w in  news_words  if not w in stops]
      
      # 6. Lemmentize 
      wordnet_lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]
      
      # 7. Stemming
      #stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in wordnet_lem ]
      
      # 8. Join the stemmed words back into one string separated by space, and return the result.
      return " ".join(wordnet_lem)

    def process_text(self, text):
      import re
      from string import punctuation
      result = text.replace('/', '').replace('\n', '')
      result = re.sub(r'[1-9]+', 'number', result)
      result = re.sub(r'(\w)(\1{2,})', r'\1', result)
      result = re.sub(r'(?x)\b(?=\w*\d)\w+\s*', '', result)
      result = ''.join(t for t in result if t not in punctuation)
      result = re.sub(r' +', ' ', result).lower().strip()
      return result

    def get_pos(self, text):
      import nltk
      from nltk import word_tokenize, pos_tag

      tokens = [nltk.word_tokenize(text)]
      postag = [nltk.pos_tag(sent) for sent in tokens][0]
      pos_list = []
      for tag in postag:
        if (tag[1].startswith('NN') or tag[1].startswith('JJ') or tag[1].startswith('VB')):
          pos_list.append(tag[0])
      return pos_list


    def get_intersection(self, pos, bias_list):
      return len(list(set(pos) & set(bias_list)))
    
    def get_wasserstein_dist(self, total_spin, total_subj, total_sens, sentiment_score):
      from scipy.stats import wasserstein_distance

      dist_from_bias = wasserstein_distance([1, 2, 0, 0.1531], [total_spin, total_subj, total_sens, sentiment_score])
      dist_from_unbias = wasserstein_distance([0, 0, 0, 0], [total_spin, total_subj, total_sens, sentiment_score])
      dist_btn = abs(dist_from_bias - dist_from_unbias)
      if (dist_from_bias < dist_btn/3):
        return 1  #highly_biased
      elif (dist_from_unbias < dist_btn/3):
        return 3 #least biased
      else:
        return 2

    
    def get_senti(self, sentence):
      import nltk.sentiment
      nltk.download('vader_lexicon')
      from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
      senti = nltk.sentiment.vader.SentimentIntensityAnalyzer()

      sentimentVector = []
      snt = senti.polarity_scores(sentence)
      sentimentVector.append(snt['neg'])
      sentimentVector.append(snt['neu'])
      sentimentVector.append(snt['pos'])
      sentimentVector.append(snt['compound'])
      return sentimentVector 

    def get_encoded_label(self, label):
      if (label == 'false' or label =='barely-true' or label =='pants-fire' or label == 'FALSE'):
        return 0
      elif ( label =='half-true' or label == 'mostly-true' or label == 'TRUE' or label =='true'):
        return 1

    def get_bias_score(self, text):
      import pandas as pd
      import numpy as np
      import pickle
      from sklearn import metrics
      from sklearn.feature_extraction.text import TfidfVectorizer
      from sklearn.feature_extraction.text import CountVectorizer
      from sklearn.model_selection import train_test_split
      from sklearn.linear_model import LogisticRegression
      from sklearn.metrics import classification_report
      from sklearn.preprocessing import StandardScaler 
      from scipy import sparse
      
        
      #creating the dataframe with our text so we can leverage the existing code
      dfrme = pd.DataFrame(index=[0], columns=['text'])
      dfrme['text'] = text

      #bias dict
      subjective_words = ['good','better', 'best', 'bad', 'worse', 'worst', 'considered', 'dangerous', 'seemingly', 'suggests', 'decrying', 'apparently', 'possibly', 'could', 'would']
      sensationalism_words = ['shocking', 'remarkable', 'rips', 'chaotic', 'lashed', 'onslaught', 'scathing', 'showdown', 'explosive','slams', 'forcing','warning','embroiled','torrent', 'desperate']
      spin_words = ['emerge','serious','refuse','crucial','high-stakes','tirade','landmark','major','critical','decrying','offend','stern','offensive','meaningful','significant','monumental','finally','concede','dodge','latest','admission','acknowledge','mock','rage','brag','lashed','scoff','frustrate','incense','erupt','rant','boast','gloat','fume',]
        
      #Creating some latent variables from the data
      dfrme['clean']     = dfrme['text'].apply(lambda x: self.cleaning(x))
      dfrme['clean']     = dfrme['clean'].apply(lambda x: self.process_text(x))
      dfrme['num_words']     = dfrme['clean'].apply(lambda x: len(x.split()))
      dfrme['pos']     = dfrme['clean'].apply(lambda x: self.get_pos(x))
      dfrme['sentiment_vector'] = dfrme['clean'].apply(lambda x: self.get_senti(x))
      dfrme['sentiment_score'] = dfrme['sentiment_vector'].apply(lambda x: x[1:][-1])
      dfrme['total_spin_bias']  = dfrme['pos'].apply(lambda x: self.get_intersection(x, spin_words))
      dfrme['total_subj_bias']  = dfrme['pos'].apply(lambda x: self.get_intersection(x, subjective_words))
      dfrme['total_sens_bias']  = dfrme['pos'].apply(lambda x: self.get_intersection(x, sensationalism_words))
      dfrme['total']     = dfrme.apply(lambda x: x.total_spin_bias + x.total_subj_bias + x.total_sens_bias, axis=1)
      dfrme['bias']      = dfrme.apply(lambda x: self.get_wasserstein_dist(x.total_spin_bias, x.total_subj_bias, x.total_sens_bias, x.sentiment_score), axis =1)

      Xtxt = dfrme['bias'].values.reshape(-1, 1)

      predicted = self.modelBias.predict(Xtxt)
      predicedProb = self.modelBias.predict_proba(Xtxt)[:,1]

      #return predicted, predicedProb
      score = 1 - float(predicedProb)
      return score  