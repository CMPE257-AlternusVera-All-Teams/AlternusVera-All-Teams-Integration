import pickle
import re
from collections import Counter
import warnings
import nltk.sentiment
import pandas as pd
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from gensim.models import Word2Vec
import numpy as np
import statistics
nltk.download('averaged_perceptron_tagger')
from gensim import corpora, models
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import scale
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
import warnings
nltk.download('punkt')
nltk.download('vader_lexicon')
warnings.filterwarnings('ignore')
nltk.download('wordnet')
nltk.download("stopwords") 
import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests
stop_words = set(stopwords.words('english'))
import os

def predictIntention(text, model):
    def cleaning(statement):
        
        # 1. Remove non-letters/Special Characters and Punctuations
        news = re.sub("[^a-zA-Z]", " ", statement)
        
        # 2. Convert to lower case.
        news =  news.lower()
        
        # 3. Tokenize.
        news_words = nltk.word_tokenize( news)
        
        # 4. Convert the stopwords list to "set" data type.
        stops = stop_words
        
        # 5. Remove stop words. 
        words = [w for w in  news_words  if not w in stops]
        
        # 6. Lemmentize 
        wordnet_lem = [ WordNetLemmatizer().lemmatize(w) for w in words ]
        
        # 7. Stemming
        stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in wordnet_lem ]
        
        # 8. Join the stemmed words back into one string separated by space, and return the result.
        return " ".join(stems)

    def clean_spell_checker(df, model):
      
        words = model.index2word
        w_rank = {}
        for i,word in enumerate(words):
            w_rank[word] = i

        WORDS = w_rank

        def words(text): return re.findall(r'\w+', text.lower())

        def P(word, N=sum(WORDS.values())): 
            "Probability of `word`."
            return - WORDS.get(word, 0)

        def correction(word): 
            "Most probable spelling correction for word."
            return max(candidates(word), key=P)

        def candidates(word): 
            "Generate possible spelling corrections for word."
            return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

        def known(words): 
            "The subset of `words` that appear in the dictionary of WORDS."
            return set(w for w in words if w in WORDS)

        def edits1(word):
            "All edits that are one edit away from `word`."
            letters    = 'abcdefghijklmnopqrstuvwxyz'
            splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
            deletes    = [L + R[1:]               for L, R in splits if R]
            transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
            replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
            inserts    = [L + c + R               for L, R in splits for c in letters]
            return set(deletes + transposes + replaces + inserts)

        def edits2(word): 
            "All edits that are two edits away from `word`."
            return (e2 for e1 in edits1(word) for e2 in edits1(e1))

        def spell_checker(text):
            all_words = re.findall(r'\w+', text.lower()) # split sentence to words
            spell_checked_text  = []
            for i in range(len(all_words)):
                spell_checked_text.append(correction(all_words[i]))
            return ' '.join(spell_checked_text)

        df['clean'] = df['clean'].apply(spell_checker)

        return df
        
    cleaned_word = []
    def clean(df, model):
        df['clean'] = df['clean'].apply(cleaning)
        df = clean_spell_checker(df, model) 
        return df


    sentiment_vector = []
    vader_polarity = []
    sentiment_score = []
    def sentiment_analysis(df):
        senti = nltk.sentiment.vader.SentimentIntensityAnalyzer()

        def print_sentiment_scores(sentence):
            snt = senti.polarity_scores(sentence)
            # print("{:-<40} \n{}".format(sentence, str(snt)))
            
        print_sentiment_scores(df['clean'][0])


        def get_vader_polarity(snt):
            if not snt:
                return None
            elif snt['neg'] > snt['pos'] and snt['neg'] > snt['neu']:
                return -1
            elif snt['pos'] > snt['neg'] and snt['pos'] > snt['neu']:
                return 1
            else:
                return 0


        def get_polarity_type(sentence):
            sentimentVector = []
            snt = senti.polarity_scores(sentence)
            sentimentVector.append(get_vader_polarity(snt))
            sentimentVector.append(snt['neg'])
            sentimentVector.append(snt['neu'])
            sentimentVector.append(snt['pos'])
            sentimentVector.append(snt['compound'])
            
            return sentimentVector

        get_pols = get_polarity_type(text)
            
        sentiment_vector = get_pols[1:]
        vader_polarity = get_pols[0]
        neg_score = get_pols[1]
        neu_score = get_pols[2]
        pos_score = get_pols[3]
        sentiment_score = get_pols[1:][-1]
        
        df['sentiment_score'] = sentiment_score
        df['vader_polarity'] = vader_polarity
        return df

    def get_sensational_score(df):
        # sensational_words = pd.read_csv('./sensational_words_dict.csv', usecols=[0], sep='\t+', header=None)
        sensational_words = pd.read_csv('/content/AlternusVera-All-Teams-Integration/trailblazers/sensational_words_dict.csv', usecols=[0], sep='\t+', header=None)
        corpus = []
        corpus.append(text)
        sensational_corpus=[]
        sensational_dictionary = ' '.join(sensational_words[0].astype(str))
        sensational_corpus.append(sensational_dictionary)
        
        # sentic_net = pd.read_csv('./senticnet5.txt', sep="\t+", header=None, usecols=[0,1,2], names = ["Token", "Polarity", "Intensity"])
        sentic_net = pd.read_csv('/content/AlternusVera-All-Teams-Integration/trailblazers/senticnet5.txt', sep="\t+", header=None, usecols=[0,1,2], names = ["Token", "Polarity", "Intensity"])
        warnings.filterwarnings("ignore")
        sentic_net = sentic_net[~sentic_net['Token'].str.contains('|'.join('_'),na=False)]
        sentic_net = sentic_net.reset_index(drop=True)
        senti_pos = sentic_net.loc[sentic_net.Polarity == "positive"]
        senti_pos = senti_pos.loc[senti_pos.Intensity > 0.90]
        dictionary = ' '.join(senti_pos.Token.astype(str))
        sensational_corpus.append(dictionary)
        
        tfidfVec = TfidfVectorizer(max_features=1000)

        train_tfidf = tfidfVec.fit_transform(df['clean'])
        max_f = train_tfidf.shape[1]
        
        tfidfVec = TfidfVectorizer(max_features=max_f)
        tfidf_corpus = tfidfVec.fit_transform(corpus)
        tf_idf_senti = tfidfVec.fit_transform(sensational_corpus)
        words = tfidfVec.get_feature_names()
        
        tfidf_corpus.toarray()

        tf_idf_senti.toarray()

        tfidfVec.vocabulary_
        
        similarity_score = []
        for i in range(len(train_tfidf.toarray())):
            similarity_score.append(1 - spatial.distance.cosine(tf_idf_senti[0].toarray(), tfidf_corpus[i].toarray()))

        df['sensational_score'] = similarity_score[0]
        return df


    def get_lda_score(df):
        data = df
        train_lda = data[['clean','index']]
        processed_docs = train_lda['clean'].map(lambda doc: doc.split(" "))
        # print(processed_docs)
        dictionary = gensim.corpora.Dictionary(processed_docs)
        # print(dictionary)
        # dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=100000)
        bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
        # print(bow_corpus)
        tfidf = models.TfidfModel(bow_corpus)
        corpus_tfidf = tfidf[bow_corpus]
        lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
        lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
        for i, data in enumerate(bow_corpus):
            for index, score in sorted(lda_model_tfidf[bow_corpus[i]], key=lambda tup: -1*tup[1]):
                df['lda_score'] = score

        return df

    
    def get_POS(df):
        stop_words = set(stopwords.words('english'))
        postags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

        for i,txt in enumerate(postags):
          df[txt]=0.00

        def getTokerns(txt):
          tokenized = sent_tokenize(txt)
          for i in tokenized:
              wordsList = nltk.word_tokenize(i)
              wordsList = [w for w in wordsList if not w in stop_words]
              tagged = nltk.pos_tag(wordsList)
              counts = Counter(tag for (word, tag) in tagged)
              total = sum(counts.values())
              a = dict((word, float(count) / total) for (word, count) in
                      counts.items())
              return a;

        for i,txt in enumerate(df['clean']):
          a = getTokerns(txt)
          for key in a:
                if key in postags:
                   df[key][i]=a[key]

        return df


    df_data = pd.DataFrame([[text,0]],columns=['clean', 'index'])
    df_data = clean(df_data, model)
    df_data = sentiment_analysis(df_data)
    df_data = get_sensational_score(df_data)
    df_data = get_lda_score(df_data)
    df_data = get_POS(df_data)
    
    df = df_data.filter(items=['lda_score','sensational_score','sentiment_score','vader_polarity',
                               'CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS',
                               'NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM',
                               'TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB'])
       
    with open('/content/AlternusVera-All-Teams-Integration/trailblazers/neural_net.pkl', 'rb') as file:  
    # with open('./neural_net.pkl', 'rb') as file:  
        neural_net_model = pickle.load(file)

    pred = neural_net_model.predict(df)
    # print(pred[0])
    
    MI_Label_map={0:'pants-fire',
                  1:'false',
                  2:'barely-true',
                  3:'half-true',
                  4:'mostly-true',
                  5:'true'}

    return MI_Label_map.get(pred[0])



Label_map={0: 'barely-true',
 1: 'false',
 2: 'full-flop',
 3: 'half-flip',
 4: 'half-true',
 5: 'mostly-true',
 6: 'pants-fire',
 7: 'true'}

speaker_map={0: 'Barack Obama',
 1: 'Chain email',
 2: 'Chris Christie',
 3: 'Donald Trump',
 4: 'Hillary Clinton',
 5: 'John McCain',
 6: 'Marco Rubio',
 7: 'Mitt Romney',
 8: 'Rick Perry',
 9: 'Scott Walker'}

def getTokerns(txt):
  tokenized = sent_tokenize(txt)
  for i in tokenized:
      wordsList = nltk.word_tokenize(i)
      wordsList = [w for w in wordsList if not w in stop_words]
      tagged = nltk.pos_tag(wordsList)
      counts = Counter(tag for (word, tag) in tagged)
      total = sum(counts.values())
      a = dict((word, float(count) / total) for (word, count) in
              counts.items())
      return a;

def predictLable(text):
  data = {'Statement':  [text]
        }
  df_test = pd.DataFrame (data, columns = ['Statement'])
  
  postags = ['CC','CD','DT','EX','FW','IN','JJ','JJR','JJS','LS','MD','NN','NNS','NNP','NNPS','PDT','POS','PRP','PRP$','RB','RBR','RBS','RP','SYM','TO','UH','VB','VBD','VBG','VBN','VBP','VBZ','WDT','WP','WP$','WRB']

  for i,txt in enumerate(postags):
    df_test[txt]=0.00;

  for i,txt in enumerate(df_test['Statement']):
    a = getTokerns(txt)
    for key in a:
        if key in postags:
          df_test[key][i]=a[key]
  
  spearker_df = df_test.filter(items=['CC', 'CD',
       'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
       'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
       'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
       'WP$', 'WRB'])
  

  cwd = os.getcwd()
  print(cwd)

  with open('/content/AlternusVera-All-Teams-Integration/trailblazers/knn_speaker_Model.pkl', 'rb') as file:  
    knn_speaker_Model = pickle.load(file)
  
  # print(spearker_df)
  sp = knn_speaker_Model.predict(spearker_df)

  res = sp[0,0];
  if res - int(res) > 0.49:
    res=int(res)+1
  else:
    res=int(res)

  src=speaker_map.get(res)

  # print(df_test)
  for key in speaker_map:
    if speaker_map[key] == src:
      df_test['Source_cat']=key
  
  with open('/content/AlternusVera-All-Teams-Integration/trailblazers/knn_truth_Model.pkl', 'rb') as file:  
    knn_truth_Model = pickle.load(file)

  source_df = df_test.filter(items=['Source_cat', 'CC', 'CD',
       'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS',
       'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP',
       'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
       'WP$', 'WRB'])
  pred=knn_truth_Model.predict(source_df)

  res_label = pred[0,0];
  if res_label - int(res_label) > 0.49:
    res_label=int(res_label)+1
  else:
    res_label=int(res_label)

  # print(pred[0,0])
  return Label_map.get(res_label)

# text='Ann and I extend our congratulations to President-elect Joe Biden and Vice President-elect Kamala Harris. We know both of them as people of good will and admirable character. We pray that God may bless them in the days and years ahead.'

# text='America, I’m honored that you have chosen me to lead our great country.The work ahead of us will be hard, but I promise you this: I will be a President for all Americans — whether you voted for me or not.I will keep the faith that you have placed in me.'
# result = predictLable(text)
# print(result)