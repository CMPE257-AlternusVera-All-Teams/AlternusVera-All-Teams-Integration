import pandas as pd
import numpy as np
import gensim
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
import nltk
import re
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')




# words = model.index2word

# w_rank = {}
# for i,word in enumerate(words):
#     w_rank[word] = i

# WORDS = w_rank

nltk.download('punkt')
nltk.download('wordnet')

def cleaning(raw_news):
    
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
    stems = [nltk.stem.SnowballStemmer('english').stem(w) for w in wordnet_lem ]
    
    # 8. Join the stemmed words back into one string separated by space, and return the result.
    return " ".join(stems)

# import re
# from collections import Counter

# def words(text): return re.findall(r'\w+', text.lower())

# def P(word, N=sum(WORDS.values())): 
#     "Probability of `word`."
#     return - WORDS.get(word, 0)

# def correction(word): 
#     "Most probable spelling correction for word."
#     return max(candidates(word), key=P)

# def candidates(word): 
#     "Generate possible spelling corrections for word."
#     return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

# def known(words): 
#     "The subset of `words` that appear in the dictionary of WORDS."
#     return set(w for w in words if w in WORDS)

# def edits1(word):
#     "All edits that are one edit away from `word`."
#     letters    = 'abcdefghijklmnopqrstuvwxyz'
#     splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
#     deletes    = [L + R[1:]               for L, R in splits if R]
#     transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
#     replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
#     inserts    = [L + c + R               for L, R in splits for c in letters]
#     return set(deletes + transposes + replaces + inserts)

# def edits2(word): 
#     "All edits that are two edits away from `word`."
#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))

# def correction(word): 
#     "Most probable spelling correction for word."
#     return max(candidates(word), key=P)

# def spell_checker(text, model):
#     all_words = re.findall(r'\w+', text.lower()) # split sentence to words
#     spell_checked_text  = []
#     for i in range(len(all_words)):
#         spell_checked_text.append(correction(all_words[i]))
#     return ' '.join(spell_checked_text)





import pickle
class EchoChamberMaster():
  '''
    Echo Chamber
  '''

  def ecoChamberScoreFinal(self,text,peopleCommented=15,numberOfShares=35,replyToComments=29,peopleLikedPost=30,spamScore=1):
    return ((100*peopleCommented+3*numberOfShares+1*replyToComments+1*peopleLikedPost)/(spamScore+1))/1000
  
  
  def predict(self,text):
    text=cleaning(text)
    r=self.ecoChamberScoreFinal(text)
    return r
    
from nltk.tokenize import word_tokenize

def get_tokens(row):
    return word_tokenize(row.lower())


def get_postags(row):
    postags = nltk.pos_tag(row)
    list_classes = list()
    for  word in postags:
        list_classes.append(word[1])
    
    return list_classes

from collections import Counter
def find_no_class(count, class_name = ""):
    total = 0
    for key in count.keys():
        if key.startswith(class_name):
            total += count[key]
            
            
    return total

def get_classes(row, grammatical_class = ""):
    count = Counter(row)
    return find_no_class(count, class_name = grammatical_class)/len(row)


class ContentStatistics():

  def __init__(self, filename):
    self.model = self.load(filename)

  def load(self, path):
    with open(path, 'rb') as file:  
       return pickle.load(file)

  def predict(self, text, model):
    length=len(text)
    text=cleaning(text)
    # text=spell_checker(text, model)
    text=get_tokens(text)
    text=get_postags(text)
    list1=[get_classes(text, "RB"),get_classes(text, "VB"),get_classes(text, "JJ"),get_classes(text, "NN"),length]
    X_test = np.array(list1).reshape(-1,5)
    return self.model.predict_proba(X_test)[0][0]