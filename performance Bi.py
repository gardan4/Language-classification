#%%
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import time
#%%
#Read in full dataset
t1 = time.time()
data = pd.read_csv('sentences.csv',
                            sep='\t',
                            encoding='utf8',
                            index_col=0,
                            names=['lang','text'])
#%%
#Filter by text length
len_cond = [True if 20<=len(s)<=200 else False for s in data['text']]
data = data[len_cond]

#Filter by text language
lang = ['deu', 'eng', 'fra', 'ita', 'por', 'spa']
data = data[data['lang'].isin(lang)]

#Select 50000 rows for each language
data_trim = pd.DataFrame(columns=['lang','text'])

for l in lang:
    lang_trim = data[data['lang'] ==l].sample(50000,random_state = 100)
    data_trim = data_trim.append(lang_trim)

#Create a random train, valid, test split
data_shuffle = data_trim.sample(frac=1)

train = data_shuffle[0:270000]
#valid = data_shuffle[210000:270000]
test = data_shuffle[270000:300000]
#%%
def get_bigrams(corpus,n_feat=200):
    """
    Returns a list of the N most common character bigrams from a list of sentences
    params
    ------------
        corpus: list of strings
        n_feat: integer
    """

    #fit the n-gram model
    vectorizer = CountVectorizer(analyzer='char',
                            ngram_range=(2, 2) ,max_features=n_feat                     )

    X = vectorizer.fit_transform(corpus)

    #Get model feature names
    feature_names = vectorizer.get_feature_names()

    return feature_names
#%%
#obtain bigrams from each language
features = {}
features_set = set()

for l in lang:

    #get corpus filtered by language
    corpus = train[train.lang==l]['text']

    #get 200 most frequent bigrams
    bigrams = get_bigrams(corpus)

    #add to dict and set
    features[l] = bigrams
    features_set.update(bigrams)


#create vocabulary list using feature set
vocab = dict()
for i,f in enumerate(features_set):
    vocab[f]=i
#%%
#train count vectoriser using vocabulary
vectorizer = CountVectorizer(analyzer='char',
                             ngram_range=(2, 2),
                            vocabulary=vocab)

#create feature matrix for training set
corpus = train['text']
X = vectorizer.fit_transform(corpus)
feature_names = vectorizer.get_feature_names()

train_feat = pd.DataFrame(data=X.toarray(),columns=feature_names)
train_feat['lang'] = list(train['lang'])

#%%
#format data for calculation
chances = {}

for l in lang:
    temp = train_feat[train_feat.lang==l].sum(axis=0)
    temp = temp.drop("lang")
    chances[l] = temp
    chances[l]["total"] = (sum(chances[l]))
#%%
 #Calculate percentage of occurences per trigram
for (key1, keyname) in chances.items():
    for (key2, valuedic) in chances[key1].items():
        chances[key1][key2] = valuedic / chances[key1]["total"]
#%% md


#%% md

#%%
def predictchance(inputSentence):
    lanChances = {}
    for l in lang:
        temp = 0
        for tri in inputSentence:
            try:
                temp += chances[l][tri]
            except:
                pass
        lanChances[l] = temp
    return lanChances
#%%
def keywithmaxval(d):
     """ a) create a list of the dict's keys and values;
         b) return the key with the max value"""
     v = list(d.values())
     k = list(d.keys())
     return k[v.index(max(v))]
#%%
#input sentence
#languages ['deu', 'eng', 'fra', 'ita', 'por', 'spa']
#accuracy[correct, wrong]
accuracy = [0,0]
for l in lang:
    inputSentence = test[test.lang==l]['text']

    for i in inputSentence:
        inputSentenceBi = get_bigrams([i])
        #print(i)
        #print(inputSentenceBi)
        lanChances = predictchance(inputSentenceBi)
        #print(f"Prediction: {keywithmaxval(lanChances)}, Actual Language: {l}")
        if keywithmaxval(lanChances) == l:
            accuracy[0] +=1
        else:
            accuracy[1] +=1

#%%
print(lanChances)
print(keywithmaxval(lanChances))
#%%
accuracyPercent = (accuracy[0] / (accuracy[1] + accuracy[0]))
print(accuracy)
print(accuracyPercent)
t2 = time.time()
print(f"elapsed time: {t2 - t1}")
#%%
