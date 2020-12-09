import numpy as np
import pandas as pd
import pickle

class ReliableSource():

  def __init__(self):
    path = "/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/ReliableSource/data.csv"
    
    
  def FeatureFinders_getSourceReliabilityScore(self, source): # return between 0 and 1, being 0 = True,  1 = Fake
    path = "/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/ReliableSource/data.csv"
    fakeNewsSites = pd.read_csv(path)
    for index, row in fakeNewsSites.iterrows():
      score = 100
      if (row['Type of site'] == 'Some fake stories'):
        score = 50
      fakeNewsSites.at[index, 'fake_score'] = score

    if (source == ""):
        return 0
    #print(source)
    d = fakeNewsSites[fakeNewsSites['Site name'].str.match(r'\b' + source + r'\b')]
    #print(d)
    if d.shape[0] > 0:
      return d.iloc[0]['fake_score']

    # if (d['fake_score'].empty):
    #     return 0
    # return int(d['fake_score'].values)
    return 0;

  def FeatureFinders_getReliabilityBySource(self,src):
    x = self.FeatureFinders_getSourceReliabilityScore(src)
    xTrain = np.array(x).reshape(-1, 1)

    readfile = open('/content/drive/My Drive/MLFall2020/the-feature-finders/AlternusVera/pickled-model/ReliableSourceLabelmodel', 'rb')
    best_clf = pickle.load(readfile)

    xPpredicted = best_clf.predict(xTrain)
    print(xPpredicted)
    xPredicedProb = best_clf.predict_proba(xTrain)[:,1]
    #xPredicedProb = best_clf.predict_proba(xTrain)
    #print(xPredicedProb)
    return 1 - float(xPredicedProb)