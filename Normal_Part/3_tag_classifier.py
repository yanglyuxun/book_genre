#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multiple labels classifier
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

#%% the classifier for training and testing the whole model

class multiclassify():
    def __init__(self,sumwds,excwds,tags):
        self._get_ids(sumwds,excwds)
        self.tags=tags
#    def _train_fiction_weights(self, sum_cl, exc_cl):
#        '''input the seperate models and train a model for fiction classification'''
#        self.fic_sum=sum_cl
#        self.fic_exc=exc_cl
    def _train_single(self, wordset, genreslist,trainid,testid,trace=True):
        models={}
        featureset = make_featureset(wordset)
        tags=self.tags
        for tag in tags:
            if trace: print(tag)
            train, test = make_data(featureset,tag,genreslist,trainid,testid)
            cl = my_clssifier(sklearn.linear_model.LogisticRegression(class_weight='balanced'))
#            cl.fit(train, test)
#            if trace: print('fit acc:',cl.acc)
            cl.pipfit(train,test)
            if trace: print('pipfit acc:',cl.pipacc)
            models[tag]=cl
        return models
    def _get_ids(self, sumwds,excwds):
        ids = [i for i in excwds]
        ids = [i for i in ids if i in sumwds]
        self.mix_trainid,self.mix_testid=set_sep(ids)
        self.sum_trainid,self.sum_testid=self._extend_ids(sumwds,self.mix_trainid,self.mix_testid)
        self.exc_trainid,self.exc_testid=self._extend_ids(excwds,self.mix_trainid,self.mix_testid)
    def _extend_ids(self,wds,trainid,testid):
        id1=trainid+testid
        idmore=[i for i in wds if i not in id1]
        trainid2,testid2=set_sep(idmore)
        return trainid+trainid2,testid+testid2

    def train_tag(self,sumwds,excwds,genreslist,trace=True):
        print('training for summary')
        self.summodel = self._train_single(sumwds,genreslist,self.sum_trainid,self.sum_testid,trace)
        print('training for excerpt')
        self.excmodel = self._train_single(excwds,genreslist,self.exc_trainid,self.exc_testid,trace)
    def use_tag_model(self,tag,source):
        if source=='sum':
            return self.summodel[tag]
        elif source=='exc':
            return self.excmodel[tag]
        else:
            print('Error!')
    def _train_mix_model_single_plot(self,tag,sumwds,excwds,genreslist):
        '''train the model to weight sum and exc for a tag'''
        ids=self.mix_trainid+self.mix_testid
        data=pd.DataFrame(index=ids,columns=['sum','exc','label'])
        s_featureset=make_featureset(sumwds)
        e_featureset=make_featureset(excwds)
        for id in ids:
            data.loc[id,'sum']=self.summodel[tag].pipclassifier.prob_classify(s_featureset[id]).prob(True)
            data.loc[id,'exc']=self.excmodel[tag].pipclassifier.prob_classify(e_featureset[id]).prob(True)
            data.loc[id,'label']= (tag in genreslist[id])
        mix_train=data.loc[self.mix_trainid]
        mix_test=data.loc[self.mix_testid]
        X_train=mix_train.iloc[:,0:2]
        X_test=mix_test.iloc[:,0:2]
        y_train=list(mix_train.iloc[:,2].values)
        y_test=list(mix_test.iloc[:,2].values)
        plt.figure()
        w=np.array(range(1,1000))/1000
        acc=lambda x:((np.sum(X_test.as_matrix()* np.array([x,1-x]),axis=1)>0.5)==(np.array(y_test)==True)).mean()
        plt.plot(w,list(map(acc,w)), label='Test')
        acc=lambda x:((np.sum(X_train.as_matrix()* np.array([x,1-x]),axis=1)>0.5)==(np.array(y_train)==True)).mean()
        plt.plot(w,list(map(acc,w)),':', label='Train')
        plt.xlabel('weight')
        plt.ylabel('accuracy')
        plt.legend()
        plt.title('Combination of two models')
    def _train_mix_model_single(self,tag,sumwds,excwds,genreslist):
        print(tag)
        ids=self.mix_trainid+self.mix_testid
        data=pd.DataFrame(index=ids,columns=['sum','exc','label'])
        s_featureset=make_featureset(sumwds)
        e_featureset=make_featureset(excwds)
        for id in ids:
            data.loc[id,'sum']=self.summodel[tag].pipclassifier.prob_classify(s_featureset[id]).prob(True)
            data.loc[id,'exc']=self.excmodel[tag].pipclassifier.prob_classify(e_featureset[id]).prob(True)
            data.loc[id,'label']= (tag in genreslist[id])
        mix_train=data.loc[self.mix_trainid]
        mix_test=data.loc[self.mix_testid]
        X_train=mix_train.iloc[:,0:2]
        X_test=mix_test.iloc[:,0:2]
        y_train=list(mix_train.iloc[:,2].values)
        y_test=list(mix_test.iloc[:,2].values)
        cl = sklearn.linear_model.LogisticRegression(class_weight='balanced')
        cl.fit(X_train, y_train)
        cl.acc = cl.score(X_test,y_test)
        print('fit acc:',cl.acc) 
        return cl
    def train_mix(self,sumwds,excwds,genreslist):
        self.mixmodel={}
        tags=self.tags
        for tag in tags:
            self.mixmodel[tag]=self._train_mix_model_single(tag,sumwds,excwds,genreslist)
        print('done.')
    def show_acc(self):
        tags=list(self.mixmodel.keys())
        acc = pd.DataFrame(index=tags,columns=['sum','exc','mix'])
        for tag in tags:
            acc.loc[tag,'sum']=self.summodel[tag].pipacc
            acc.loc[tag,'exc']=self.excmodel[tag].pipacc
            acc.loc[tag,'mix']=self.mixmodel[tag].acc
        self.acc_table = acc
        return acc
    def _calc_fscore_single(self,tag,sumwds,excwds,genreslist):
        print(tag)
        ids=self.mix_testid
        data=pd.DataFrame(index=ids,columns=['sum','exc','mix','label'])
        s_featureset=make_featureset(sumwds)
        e_featureset=make_featureset(excwds)
        for id in ids:
            data.loc[id,'sum']=self.summodel[tag].pipclassifier.prob_classify(s_featureset[id]).prob(True)
            data.loc[id,'exc']=self.excmodel[tag].pipclassifier.prob_classify(e_featureset[id]).prob(True)
            data.loc[id,'label']= (tag in genreslist[id])
        data.loc[:,'mix']=self.mixmodel[tag].predict(data.loc[:,['sum','exc']])
        data.loc[:,'sum']=(data.loc[:,'sum']>0.5)
        data.loc[:,'exc']=(data.loc[:,'exc']>0.5)
        fun=sklearn.metrics.precision_recall_fscore_support
        def fun(y_true,y_pred):
            y_true1,y_pred1 = list(y_true.values),list(y_pred.values)
            return sklearn.metrics.precision_recall_fscore_support(y_true1,y_pred1,labels=[True,False])
        fsum=fun(data['label'],data['sum'])
        fexc=fun(data['label'],data['exc'])
        fmix=fun(data['label'],data['mix'])
        return fsum,fexc,fmix
    def show_fscore(self,sumwds,excwds,genreslist):
        tags = self.tags
        scores = {}
        for tag in tags:
            fsum,fexc,fmix = self._calc_fscore_single(tag,sumwds,excwds,genreslist)
            scores[tag] = {'sum':fsum,'exc':fexc,'mix':fmix}
        self.scores=scores
        return scores
    def get_f1score(self):
        scores=self.scores
        tags=self.tags
        acc = pd.DataFrame(index=tags,columns=['sum','exc','mix'])
        for tag in tags:
            for c in acc.columns:
                acc.loc[tag,c]=scores[tag][c][2].mean()
        self.acc_table = acc
        return acc
    def save(self):
        save(self.__dict__,'tag_models.pc')
    def load(self):
        self.__dict__.update(load('tag_models.pc'))

features = lambda words: {w:n for w,n in Counter(words).items()}
def make_featureset(wordset):
    '''make the feature set from a words dictionary'''
    return {c:features(d) for c,d in wordset.items()}
def make_data(featureset,tag,genreslist,trainid,testid):
    '''after make_featureset, make the data for nltk'''
    train=[(featureset[c], tag in genreslist[c]) for c in trainid]
    test=[(featureset[c], tag in genreslist[c]) for c in testid]
    return train,test


def set_sep(sets0,test_frac=0.1):
    sets = sets0.copy() 
    random.seed(777)
    random.shuffle(sets)
    n = int(test_frac*len(sets))
    return sets[n:], sets[:n]
class my_clssifier():
    def __init__(self, skclssif):
        self.skclssif=skclssif
        self.classif=nltk.classify.scikitlearn.SklearnClassifier(self.skclssif)
#        try:
#            self.classif._clf.set_params(n_jobs=-1)
#        except:
#            pass
        self.pip=Pipeline([('tfidf', sklearn.feature_extraction.text.TfidfTransformer()),
                     #('chi2', sklearn.feature_selection.SelectKBest(sklearn.feature_selection.chi2, k=1000)),
                     ('NB',skclssif)])
        self.pipclassif=nltk.classify.scikitlearn.SklearnClassifier(self.pip)
    def fit(self, train, test):
        self.classifier=self.classif.train(train)
        self.acc=(nltk.classify.accuracy(self.classifier, test))
    def pipfit(self, train, test):
        self.pipclassifier=self.pipclassif.train(train)
        self.pipacc=(nltk.classify.accuracy(self.pipclassifier, test))

#%% training

tags=set()
for id in genreslist:
    tags.update(genreslist[id])
tags.remove('Fiction')
tags.remove('Nonfiction')

tag_cl = multiclassify(sumwds,excwds,tags)
tag_cl.train_tag(sumwds,excwds,genreslist)
tag_cl.train_mix(sumwds,excwds,genreslist)
tag_cl.save()



#%% read the model

tags=set()
for id in genreslist:
    tags.update(genreslist[id])
tags.remove('Fiction')
tags.remove('Nonfiction')
tag_cl = multiclassify(sumwds,excwds,tags)
tag_cl.load()
tag_cl._train_mix_model_single_plot(list(tags)[0],sumwds,excwds,genreslist)


acctable = tag_cl.show_acc()
acctable.mean()

# tag_cl.load()
# sumtestid=tag_cl.sumtestid

# add sample rate
acctable['per']=0
for tag in tags:
    acctable.loc[tag,'per']=np.mean([tag in genreslist[id] for id in genreslist])
    
#fscores=tag_cl.show_fscore(sumwds,excwds,genreslist)
f1score=tag_cl.get_f1score()
f1score['per']=0
for tag in tags:
    f1score.loc[tag,'per']=np.mean([tag in genreslist[id] for id in genreslist])
    