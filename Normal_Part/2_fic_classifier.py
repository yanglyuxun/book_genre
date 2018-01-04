#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:51:00 2017

@author: ylx
"""
import nltk
from nltk.stem import WordNetLemmatizer
import zipfile
import pandas as pd
import numpy as np
import pickle
from collections import Counter
import gzip
import random
import sklearn
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.metrics import *
from sklearn.pipeline import Pipeline

def save(obj, filename, protocol=pickle.DEFAULT_PROTOCOL):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

booklist=load('booklist.pc')
summarylist,excerptlist,genreslist=load('sum_exc_gen.pc')
sumwds,excwds = load('sum_exc_words.pc')
#%% functions and data
def features(words):
    return {w:n for w,n in Counter(words).items()}
def set_sep(sets0,test_frac=0.1):
    sets = sets0.copy() 
    random.seed(777)
    random.shuffle(sets)
    n = int(test_frac*len(sets))
    return sets[n:], sets[:n]
def wcloud(freq,size=(2000,1000)):
    wordcloud = WordCloud(background_color='white',
                          width=size[0],
                          height=size[1]).generate_from_frequencies(freq)
    plt.figure(figsize=(size[0]/100,size[1]/100))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()
def most_commen(cl,label,n=100):
    cpdist = cl._feature_probdist
    ans = {}
    for (fname, fval) in cl.most_informative_features(n):
        def labelprob(l):
            return cpdist[l, fname].prob(fval)
        labels = sorted([l for l in cl._labels
                         if fval in cpdist[l, fname].samples()],
                        key=labelprob)
        if len(labels) == 1:
            continue
        l0 = labels[0]
        l1 = labels[-1]
        if cpdist[l0, fname].prob(fval) != 0:
            ratio = cpdist[l1, fname].prob(fval) / cpdist[l0, fname].prob(fval)
            if l0==label:
                ratio=1/ratio if ratio!=0 else 0
            ans.update({fname:ratio})
    return ans

# a wrapper 2
class my_clssifier():
    def __init__(self, skclssif):
        self.skclssif=skclssif
        self.classif=nltk.classify.scikitlearn.SklearnClassifier(self.skclssif)
        try:
            self.classif._clf.set_params(n_jobs=-1)
        except:
            pass
        self.pip=Pipeline([('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
                     #('chi2', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=1000)),
                     ('NB',skclssif)])
        self.pipclassif=nltk.classify.scikitlearn.SklearnClassifier(self.pip)
    def fit(self, train, test):
        self.classifier=self.classif.train(train)
        print(nltk.classify.accuracy(self.classifier, test))
    def pipfit(self, train, test):
        self.pipclassifier=self.pipclassif.train(train)
        print(nltk.classify.accuracy(self.pipclassifier, test))
        
# not good!
#def measure(cl,test):
#    ref=[i[1] for i in sum_test]
#    testdata=[classifier.classify(i[0]) for i in sum_test]
#    refset=set(ref)
#    testdataset=set(testdata)
#    def f(fun):
#        return fun(ref,testdata)
#    def f2(fun):
#        return fun(refset,testdataset)
#    return f(accuracy),f2(precision),f2(recall),f2(f_measure)

# make data
# make data ids
ids = [i for i in excwds]
ids = [i for i in ids if i in sumwds]
ids = [i for i in ids if not pd.isnull(booklist.genre0.loc[i])]
trainid,testid=set_sep(ids)
def extend_ids(wds,trainid,testid):
    id1=trainid+testid
    idmore=[i for i in wds if i not in id1]
    idmore=[i for i in idmore if not pd.isnull(booklist.genre0.loc[i])]
    trainid2,testid2=set_sep(idmore)
    return trainid+trainid2,testid+testid2
sum_trainid,sum_testid=extend_ids(sumwds,trainid,testid)
exc_trainid,exc_testid=extend_ids(excwds,trainid,testid)

# summary
sum_train = [(features(sumwds[c]), booklist.genre0.loc[c]) for c in sum_trainid]
sum_test = [(features(sumwds[c]), booklist.genre0.loc[c]) for c in sum_testid]
# excerpt
exc_train = [(features(excwds[c]), booklist.genre0.loc[c]) for c in exc_trainid]
exc_test = [(features(excwds[c]), booklist.genre0.loc[c]) for c in exc_testid]


#%% naive Bayes and word cloud
classif = nltk.NaiveBayesClassifier.train
## summary
classifier = classif(sum_train)
print(nltk.classify.accuracy(classifier, sum_test))
# 0.6057007125890737
classifier.show_most_informative_features()
# word cloud
wcloud(most_commen(classifier,'Fiction'))
plt.savefig('figure/sum_fic.jpg')
wcloud(most_commen(classifier,'Nonfiction'))
plt.savefig('figure/sum_nonfic.jpg')

## excerpt
classifier = classif(exc_train)
print(nltk.classify.accuracy(classifier, exc_test))
# 0.8547215496368039
classifier.show_most_informative_features()
# word cloud
wcloud(most_commen(classifier,'Fiction'))
plt.savefig('figure/exc_fic.jpg')
wcloud(most_commen(classifier,'Nonfiction'))
plt.savefig('figure/exc_nonfic.jpg')


#%% naive Bayes from sklearn
from sklearn.naive_bayes import MultinomialNB
cl=my_clssifier(sklearn.naive_bayes.MultinomialNB())
cl.fit(sum_train, sum_test)# 0.6073546856465006
cl.pipfit(sum_train, sum_test)# 0.604982206405694
cl.fit(exc_train, exc_test)# 0.8357487922705314
cl.pipfit(exc_train, exc_test)# 0.6328502415458938
0.6037959667852907
0.604982206405694
0.8357487922705314
0.6328502415458938

#%% Max Ent from sklearn
cl=my_clssifier(sklearn.linear_model.LogisticRegression(class_weight='balanced'))
cl.fit(sum_train, sum_test)# 0.5516014234875445
cl.pipfit(sum_train, sum_test)# 0.6073546856465006
cl.fit(exc_train, exc_test)# # 0.8719806763285024
cl.pipfit(exc_train, exc_test)# 0.8864734299516909
0.5605700712589073
0.5783847980997625
0.8886198547215496
0.8958837772397095
save(cl,'maxent_exc.model')

#%% SVM
cl=my_clssifier(sklearn.svm.SVC())
cl.fit(sum_train, sum_test)# 0.5966785290628707
cl.pipfit(sum_train, sum_test)# 0.5966785290628707
cl.fit(exc_train, exc_test)# 0.7753623188405797
cl.pipfit(exc_train, exc_test)# 0.6328502415458938
0.5966785290628707
0.5966785290628707
0.7753623188405797
0.6328502415458938

#%% RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
cl=my_clssifier(sklearn.ensemble.RandomForestClassifier())
cl.fit(sum_train, sum_test)#
cl.pipfit(sum_train, sum_test)#
cl.fit(exc_train, exc_test)# 
cl.pipfit(exc_train, exc_test)#
0.5776986951364176
0.5705812574139977
0.8405797101449275
0.785024154589372

#%% AdaBoostClassifier
cl=my_clssifier(sklearn.ensemble.AdaBoostClassifier())
cl.fit(sum_train, sum_test)#
cl.pipfit(sum_train, sum_test)#
cl.fit(exc_train, exc_test)# 
cl.pipfit(exc_train, exc_test)#
0.6073546856465006
0.604982206405694
0.8478260869565217
0.8695652173913043


#%% GradientBoostingClassifier
cl=my_clssifier(sklearn.ensemble.GradientBoostingClassifier())
cl.fit(sum_train, sum_test)#
cl.pipfit(sum_train, sum_test)#
cl.fit(exc_train, exc_test)# 
cl.pipfit(exc_train, exc_test)#
0.6061684460260973
0.6073546856465006
0.8913043478260869
0.8840579710144928

#%% BaggingClassifier
cl=my_clssifier(sklearn.ensemble.BaggingClassifier())
cl.fit(sum_train, sum_test)#
cl.pipfit(sum_train, sum_test)#
cl.fit(exc_train, exc_test)# 
cl.pipfit(exc_train, exc_test)#
0.5563463819691578
0.5575326215895611
0.8333333333333334
0.8429951690821256

#%% KNeighborsClassifier
cl=my_clssifier(sklearn.neighbors.KNeighborsClassifier())
cl.fit(sum_train, sum_test)#
cl.pipfit(sum_train, sum_test)#
cl.fit(exc_train, exc_test)# 
cl.pipfit(exc_train, exc_test)#
0.575326215895611
0.5670225385527876
0.7246376811594203
## Error

#%% RadiusNeighborsClassifier ( Error!)

#%% MLPClassifier
from sklearn.neural_network import MLPClassifier
cl=my_clssifier(sklearn.neural_network.MLPClassifier())
cl.fit(sum_train, sum_test)#
cl.pipfit(sum_train, sum_test)#
cl.fit(exc_train, exc_test)# 
cl.pipfit(exc_train, exc_test)#
0.5599051008303677
0.5480427046263345
0.8816425120772947
0.8840579710144928