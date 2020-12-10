# -*- coding: utf-8 -*-
# load required libraries and read data
import pandas as pd
import numpy as np
import gensim
import string
import re
import nltk
import math
import pickle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim.models import word2vec
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer 
from scipy import spatial
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import spacy
from string import punctuation
from scipy.stats import wasserstein_distance
import tensorflow as tf
import ktrain
import textstat


"""AlternusVera_Topic_LDA_Sprint4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gxwoiBlbXvC4Qehnx07hlIG_ju37H_dh
"""

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
    # Gensim
    import gensim
    import gensim.corpora as corpora
    from gensim.utils import simple_preprocess
    from gensim.models import CoherenceModel
    #load the model

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

    #append dataset with sentiment label and normalized encoded value
    sentiment_score = pd.DataFrame(sentiment_score_dt)
    df_testing['sentiment'] = sentiment_score['sentiment_label']
    df_testing['sentiment_encode'] = sentiment_score['sentiment_label_encode']
    df_testing['Keywords'] = dt_dominant_topic['Keywords']
    df_testing['Dominant_Topic'] = dt_dominant_topic['Dominant_Topic']
    df_testing['Topic_Score'] = dt_dominant_topic['Topic_Score']

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
      from nltk.stem.wordnet import WordNetLemmatizer
      lemmatizer = nltk.WordNetLemmatizer()
      from nltk.corpus import stopwords
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



"""girlswhocode_political_affiliation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/174UwkYutB7AJvFMiEBY0DbC0eEiPfPOd
"""

class Girlswhocode_PoliticalAfiiliation:
  nltk.download('stopwords')

  def __init__(self, filenameModelPA, partyAffiliationDict, newDict, train): 
      self.modelPA = self.__load(filenameModelPA)
      self.partyAffiliationDict = partyAffiliationDict 
      self.newDict = newDict 
      self.train = train


  def __load(self, path):
      with open(path, 'rb') as file:
          return pickle.load(file) 
         

  #label encoding
  def encode_news_type(self, input_label):
      true_labels = ['original','true','mostly-true','half-true']
      false_labels = ['barely-true','false','pants-fire']
      if input_label in true_labels:
          return 1
      else:
          return 0

  #method to remove punctuations from textual data
  def remove_punctuation(self, text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

 

  #Remove stop words
  def remove_stopwords(self, text):
      sw = stopwords.words('english')
      stemmer = SnowballStemmer("english")
      text = [word.lower() for word in text.split() if word.lower() not in sw]
      return " ".join(text)

  #Lemmetize and pos tagging
  def lemmatize_stemming(self, text):
      sw = stopwords.words('english')
      stemmer = SnowballStemmer("english")
      return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

  #Stemming
  def stemming(self, text): 
      sw = stopwords.words('english')
      stemmer = SnowballStemmer("english")
      text = [stemmer.stem(word) for word in text.split()]
      return " ".join(text)

  def text_preprocess(self, df):
    #encode labels
    #df['encoded_label'] = df.apply(lambda row: encode_news_type(row['label']), axis=1)
    #convert to lower case
    df['headline_text'] = df['headline_text'].str.lower()
    #remove stop words
    df['headline_text'] = df['headline_text'].apply(self.remove_stopwords)
    #spell check
    #df['headline_text'] = df['headline_text'].apply(spell_checker)
    #Lemmetize
    df['headline_text'] = df['headline_text'].apply(self.lemmatize_stemming)
    #stemming
    df['headline_text'] = df['headline_text'].apply(self.stemming)
    #remove punctuation
    df['headline_text'] = df['headline_text'].apply(self.remove_punctuation)
    #remove less than 3 letter words
    df['headline_text']  = df.headline_text.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))
    return df[['headline_text',  'partyaffiliation']]

  def train_preprocess(self, df):
    #encode labels
    df['encoded_label'] = df.apply(lambda row: self.encode_news_type(row['label']), axis=1)
    #convert to lower case
    df['headline_text'] = df['headline_text'].str.lower()
    #remove stop words
    df['headline_text'] = df['headline_text'].apply(self.remove_stopwords)
    #spell check
    #df['headline_text'] = df['headline_text'].apply(spell_checker)
    #Lemmetize
    df['headline_text'] = df['headline_text'].apply(self.lemmatize_stemming)
    #stemming
    df['headline_text'] = df['headline_text'].apply(self.stemming)
    #remove punctuation
    df['headline_text'] = df['headline_text'].apply(self.remove_punctuation)
    #remove less than 3 letter words
    df['headline_text']  = df.headline_text.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))
    return df[['headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo', 'partyaffiliation', 'context', 'encoded_label']]

    

  def encode_party_affiliation_type(self, input_label):
      labels = ['democrat','republican','independent']
      if input_label not in labels:
        return str('other')
      else:
        return input_label

  def convert_partyaffiliation_category(self, df):
      partyaffiliation_dict = {'independent':4, 'other':3, 'democrat':2, 'republican':1}
      pa = []
      for index,row in df.iterrows():
        pa.append(partyaffiliation_dict[row['partyaffiliation']])
      return pa

  def tag_headline(self, df, label):
      tagged_text = []
      for index, row in df.iterrows():
        tagged_text.append(TaggedDocument(words=word_tokenize(row['headline_text']), tags=[row[label]]))
      return tagged_text

  def get_issues_vector(self, df_testing,issues):
      text_test =  ' '.join(issues)
      vector = []
      for row in df_testing.headline_text:
        # tokenization 
        head_line = str(row)
        text_list = word_tokenize(text_test)  
        headline_list = word_tokenize(head_line)
      
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
          vector.append(cosine)
      return vector

  def computeTF(self, df, dictionary):
    TF = []
    dict_words = dictionary['word'].unique()
    for index, row in df.iterrows():
        row_freq = []
        words = row['headline_text'].split()
        for i in range(len(dict_words)):
            frequency = float(words.count(dict_words[i])/len(dict_words))
            row_freq.append(frequency)
        TF.append(row_freq)
    return TF

  #Calculate IDF for the dictionary
  import math
  def computeIDF(self, df, dictionary):
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
    return IDF

  #Calculate TF-IDF for each headline text based on the dictionary created
  def computeTFIDF(self, TF, IDF):
    TFIDF = []
    IDF = np.asarray(IDF)
    for j in TF:
        tfidf = np.asarray(j) * IDF.T
        TFIDF.append(tfidf)
    return TFIDF

  def sentiment_analyzer_scores(self, df):
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
  def get_word_tokens(self, text):
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


  def identify_topic_number_score(self, text,df_train):
    documents = df_train[['headline_text']]
    processed_docs = documents['headline_text'].map(self.get_word_tokens)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    bow_vector = dictionary.doc2bow(self.get_word_tokens(text))
    topic_number , topic_score = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])[0]
    return pd.Series([topic_number, topic_score])

  """def get_political_affiliation_test_vector(df, doc2vec_model_pa_test):
    #doc2vec_model_pa_test = Doc2Vec(documents = tagged_pa_text_test, dm=0, num_features=500, min_count=2, size=20, window=4)
    pa_test = []
    for i in range(len(df['partyaffiliation_encode'])):
        pa_value = df['partyaffiliation_encode'][i]
        pa = doc2vec_model_pa_test[pa_value] + df['sentiment_encode'][i] + df['topic_score'][i] +df['republican_vector'][i]+df['democrats_vector'][i]+df['liberterian_vector'][i]
        pa_test.append(pa)
    return pa_test"""

  def predict(self, model, X_test):
    y_pred = model.predict(X_test)
    predicted_proba = model.predict_proba(X_test)[:,1]
    return y_pred, predicted_proba

  def DATAMINERS_getPartyAffiliationScore(self, headline, party):
      test = [[headline, party]]

      #creating the dataframe
      df_testing = pd.DataFrame(test,columns=['headline_text','partyaffiliation'])

      #preprocessing of the text
      df_testing = self.text_preprocess(df_testing)

      #encoding partyaffiliation
      pa_encode_test = pd.DataFrame(self.convert_partyaffiliation_category(df_testing))
      df_testing['partyaffiliation_encode'] = pa_encode_test

      #issues related to democrats,republicans and libertarian
      democrats_issues = ['minimum wage', 'health care','education','environment','renewable energy','fossil fuels']
      libertarian_issues = ['taxes','economy','civil liberties','crime and justice','foreign policy','healthcare','gun ownership','war on drugs','immigration']
      republican_issues =['Abortion and embryonic stem cell research','Civil rights','Gun ownership','Drugs','Education','Military service','Anti-discrimination laws']

      df_testing['republican_vector'] = self.get_issues_vector(df_testing,republican_issues)
      
      df_testing['democrats_vector'] = self.get_issues_vector(df_testing,democrats_issues)
      
      df_testing['liberterian_vector'] = self.get_issues_vector(df_testing,libertarian_issues)

     #party_affiliation_dict = pd.read_csv('/content/drive/MyDrive/MLFall2020/girlswhocode/datasets/Alternusvera_dataset/party_affiliaiton_dict.csv')
      party_affiliation_dict = pd.read_csv(self.partyAffiliationDict)
      TF_scores = self.computeTF(df_testing, party_affiliation_dict)
      IDF_scores = self.computeIDF(df_testing, party_affiliation_dict)
      TFIDF_scores = self.computeTFIDF(TF_scores, IDF_scores)

      #append tfidf score to the dataframe
      tfidf_test = []
      for i in range(len(TFIDF_scores)):
        tfidf_test.append(max(TFIDF_scores[i]))

      df_testing['tfidf'] = tfidf_test

      #tfidf wrt issues, get the issues dictionary which has the frequency of all the issues
      #issues_dict = pd.read_csv('/content/drive/MyDrive/MLFall2020/girlswhocode/datasets/Alternusvera_dataset/new_dict.csv')
      issues_dict = pd.read_csv(self.newDict)

      TFissues_scores = self.computeTF(df_testing, issues_dict)
      IDFissues_scores = self.computeIDF(df_testing, issues_dict)
      TFIDFissues_scores = self.computeTFIDF(TFissues_scores, IDFissues_scores)

      #append issues score to the dataframe
      tfidf_issues = []
      for i in range(len(TFIDFissues_scores)):
        tfidf_issues.append(max(TFIDFissues_scores[i]))

      df_testing['issues_score'] = tfidf_issues


      #get sentiment score related to party
      sentiment_score_input = self.sentiment_analyzer_scores(df_testing)
      sentiment_score_input = pd.DataFrame(sentiment_score_input)
      df_testing['sentiment'] = sentiment_score_input['senti_label']
      df_testing['sentiment_encode'] = sentiment_score_input['senti_label_encode']

      #load train dataset to get the dictionary of issues to calculate the topic score of each headline text
      colnames = ['jsonid', 'label', 'headline_text', 'subject', 'speaker', 'speakerjobtitle', 'stateinfo','partyaffiliation', 'barelytruecounts', 'falsecounts','halftruecounts','mostlytrueocunts','pantsonfirecounts','context']
    #df_train = pd.read_csv('/content/drive/MyDrive/MLFall2020/girlswhocode/datasets/Alternusvera_dataset/train.tsv', sep='\t', names = colnames,error_bad_lines=False)
      df_train = pd.read_csv(self.train, sep='\t', names = colnames,error_bad_lines=False)
      #preprocess the df_train headline_text
      df_train = self.train_preprocess(df_train)

      


      #get topic score related to party
      df_testing[['topic_number','topic_score']] = df_testing.apply(lambda row: self.identify_topic_number_score(row['headline_text'],df_train), axis=1)

      #get the tagged text fot the headline
      tagged_pa_text_test = self.tag_headline(df_testing, 'partyaffiliation_encode')
      X_test = []
      for i in range(len(df_testing['partyaffiliation_encode'])):
          pa_value = df_testing['partyaffiliation_encode'][i]
          pa = df_testing['sentiment_encode'].iloc[i] + df_testing['topic_score'].iloc[i] +df_testing['republican_vector'].iloc[i]+df_testing['democrats_vector'].iloc[i]+df_testing['liberterian_vector'].iloc[i]+df_testing['tfidf'].iloc[i]+df_testing['issues_score'].iloc[i]
          X_test.append(pa)
        
      X_test = np.array(X_test)
      X_test = X_test.reshape(-1,1)
      #load the saved model
      # fakenews_classifier = pickle.load(open('/content/drive/MyDrive/MLFall2020/girlswhocode/models/political_affiliation_model.sav', 'rb'))

      #predict the test input
    #  binary_value_predicted, predicted_proba = self.predict(fakenews_classifier, X_test)
      binary_value_predicted, predicted_proba = self.predict(self.modelPA, X_test)
  
      return (1 - float(predicted_proba))



"""ToxicityClass.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1f2vn2FviqJRxECq6yjDiBAdiydTxu-Pp
"""

# -*- coding: utf-8 -*-
"""Copy of ToxicityClass.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/13g0cwrrcYHD2FxoRNyKw1iT_-NUs140E
"""

class GirlsWhoCode_Toxicity:
  """def encodeLabel(df):
    df.label[df.label == 'FALSE'] = 0
    df.label[df.label == 'half-true'] = 1
    df.label[df.label == 'mostly-true'] = 1
    df.label[df.label == 'TRUE'] = 1
    df.label[df.label == 'barely-true'] = 0
    df.label[df.label == 'pants-fire'] = 0
    return df"""

  def __init__(self, filenameModelTX, pathModelBR): 
      self.modelTX = self.__load(filenameModelTX)
      self.modelBR = ktrain.load_predictor(pathModelBR)


  def __load(self, path):
      with open(path, 'rb') as file:
          return pickle.load(file) 
         
  def cleaning(self, raw_news):
    import nltk
    from nltk.stem.wordnet import WordNetLemmatizer
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    import nltk

    # 1. Remove non-letters/Special Characters and Punctuations
    news = re.sub("[^a-zA-Z]", " ", raw_news)

    # 2. Convert to lower case.
    news =  news.lower()

    # 3. Tokenize.
    news_words = nltk.word_tokenize( news)

    # 4. Convert the stopwords list to "set" data type.
    stops = set(nltk.corpus.stopwords.words("english"))

    # 5. Remove stop words.
    words = [w for w in  news_words  if not w in stops]

    # 6. Lemmentize
    wordnet_lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]

    # 7. Stemming
    stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in wordnet_lem ]

    # 8. Join the stemmed words back into one string separated by space, and return the result.
    return " ".join(stems)

  def getDataFrameWithToxicity(self, df):
    # toxicityPredictor = ktrain.load_predictor('/content/drive/MyDrive/MLFall2020/girlswhocode/models/BERTOnLiarLiar')
    news = df.loc[:,'headline_text']
    dt = news.values
    df['toxicity'] =  self.modelBR.predict(dt)
    return df

  def encodeToxicity(self, df):
    df.toxicity[df.toxicity == 'non-toxic'] = 2
    df.toxicity[df.toxicity == 'toxic'] = 1
    return df

  def getSentiment(self, df):
    sid_obj = SentimentIntensityAnalyzer()
    for i in range(len(df)) :
      sentiment_dict = sid_obj.polarity_scores(df.loc[i,"headline_text"])
      df.loc[i,"POSITIVE"] =sentiment_dict['pos']
      df.loc[i,"NEUTRAL"] =sentiment_dict['neu']
      df.loc[i,"NEGATIVE"] =sentiment_dict['neg']
      df.loc[i,"COMPOUND"] = sentiment_dict['compound']
    for i in range(len(df)) :
      if  df.loc[i,"COMPOUND"] >=0.05:
        sentiment = 'Positive'
      elif df.loc[i,"COMPOUND"] > -0.05 and df.loc[i,"COMPOUND"] < 0.05:
        sentiment = 'Neutral'
      else :
        sentiment = 'Negative'
    df.loc[i,"SENTIMENT"] = sentiment
    df =df.drop(['POSITIVE','NEUTRAL','NEGATIVE','COMPOUND'],axis=1)
    df.SENTIMENT[df.SENTIMENT == 'Positive'] = 3
    df.SENTIMENT[df.SENTIMENT == 'Negative'] = 2
    df.SENTIMENT[df.SENTIMENT == 'Neutral'] = 1
    return df

  def get_word_tokens(self, text):
    import gensim
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if len(token) > 3:
            result.append(token)
    return result

  def identify_topic_number_score(self, df,text):
    import gensim
    documents = df[['headline_text']]
    processed_docs = documents['headline_text'].map(self.get_word_tokens)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
    bow_vector = dictionary.doc2bow(self.get_word_tokens(text))
    topic_number , topic_score = sorted(lda_model[bow_vector], key=lambda tup: -1*tup[1])[0]

    return pd.Series([topic_number, topic_score])


  #Scalar Vector for testing dataset
  def testing_dataset_vector(self, df_testing):
    vec_testing = []
    for i in range(len(df_testing['toxicity'])):
        vec = df_testing['sentiment_encode'].iloc[i] + df_testing['topic_score'].iloc[i]
        vec_testing.append(vec)
    return vec_testing


  def sentiment_analyzer_scores(self, df):
    analyser = SentimentIntensityAnalyzer()
    sentiment_score = []
    sentiment_labels = {0:'negative', 1:'positive', 2:'neutral'}
    for index,row in df.iterrows():
        score = analyser.polarity_scores(row['headline_text'])
        values = [score['neg'], score['pos'], score['neu']]
        max_index = values.index(max(values))
        data = {'sentiment_score':score, 'sentiment_label':sentiment_labels[max_index], 'sentiment_label_encode': 1+math.log(max_index+1)}
        sentiment_score.append(data)
    return sentiment_score


  def get_toxicity__vector(self, df):
    from gensim.models.doc2vec import TaggedDocument
    from gensim.models import Doc2Vec
    import nltk
    from nltk import word_tokenize
    nltk.download('punkt')
    tagged_text = []
    for index, row in df.iterrows():
        tagged_text.append(TaggedDocument(words=word_tokenize(row['headline_text']), tags=[row['label']]))

    tagged_pa_text_test = tagged_text

    #train doc2vec model
    doc2vec_model_pa_test = Doc2Vec(documents = tagged_pa_text_test, dm=0, num_features=500, min_count=2, size=20, window=4)
    y_pa_test, X_pa_test = create_vector_for_learning(doc2vec_model_pa_test, tagged_pa_text_test)
    pa_test = []
    for i in range(len(df['toxicity'])):
        pa_value = df['toxicity'][i]
        pa = doc2vec_model_pa_test[pa_value] + df['SENTIMENT'][i] + df['topic_score'][i]
        pa_test.append(pa)
    return pa_test

  def tag_headline(self, df, label):
    import gensim
    from gensim.models.doc2vec import TaggedDocument
    import nltk
    from nltk import word_tokenize
    nltk.download('punkt')
    tagged_text = []
    for index, row in df.iterrows():
        tagged_text.append(TaggedDocument(words=word_tokenize(row['headline_text']), tags=[row[label]]))
    return tagged_text

  def getReadability(self, df):
    df['ARI'] = df.headline_text.apply(lambda x:textstat.automated_readability_index(x))
    df['DCR'] = df.headline_text.apply(lambda x:textstat.dale_chall_readability_score(x))
    df['TS'] = df.headline_text.apply(lambda x:textstat.text_standard(x,float_output =True))
    return df


  def getToxicityScore(self, headline):
    #converting the text and the label into a dataframe
    cols = [[headline]]
    df_testing = pd.DataFrame(cols,columns=['headline_text'])
    df_testing.head()


    #cleanup the text
    df_testing['headline_text'] = df_testing["headline_text"].apply(self.cleaning)

    #Next to predict the toxicity we will use the BERT model that was trained on the liar liar dataset
    df_testing = self.getDataFrameWithToxicity(df_testing)


    #encoding the toxicity
    df_testing = self.encodeToxicity(df_testing)

    sentiment_score_dt = self.sentiment_analyzer_scores(df_testing)
    sentiment_score = pd.DataFrame(sentiment_score_dt)

    df_testing['sentiment'] = sentiment_score['sentiment_label']
    df_testing['sentiment_encode'] = sentiment_score['sentiment_label_encode']

    #Topic modelling
    import gensim
    from gensim.models.doc2vec import TaggedDocument

    from gensim.models.doc2vec import TaggedDocument
    from gensim.models import Doc2Vec
    df_testing[['topic_number','topic_score']] = df_testing.apply(lambda row: self.identify_topic_number_score(df_testing,row['headline_text']), axis=1)
    df_testing = self.getReadability(df_testing)
    X_test = np.array(self.testing_dataset_vector(df_testing))
    X_test =  X_test.reshape(-1, 1)
    #y_test = df_testing['label']

    #fake_news_classifier = pickle.load(open('/content/drive/MyDrive/MLFall2020/girlswhocode/models/toxicityModel.sav', 'rb'))
    cols = list(df_testing.columns)
    cols.remove('headline_text')
    #cols.remove('label')
    cols.remove('sentiment')
    predicted = self.modelTX.predict(df_testing[cols])
    predicedProb = self.modelTX.predict_proba(df_testing[cols])[:,1]

    score = 1 - float(predicedProb)
    return score
      
