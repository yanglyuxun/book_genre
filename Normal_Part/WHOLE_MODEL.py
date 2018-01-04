#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The whole trained model for fiction genre classification
"""
#%%
import numpy as np
import pandas as pd
#import math
#from sklearn.feature_extraction import DictVectorizer
from sparse_class import sparse
#from functools import reduce
from nltk.probability import DictionaryProbDist
import nltk
from nltk.stem import WordNetLemmatizer
#import zipfile
import pickle
from collections import Counter
import gzip
import random
import sklearn
#from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.metrics import *
from sklearn.pipeline import Pipeline
#from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def save(obj, filename, protocol=pickle.DEFAULT_PROTOCOL):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)

def load(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
    return loaded_object

class my_MaxEnt():
    'A maximum entropy classifier'
    def __init__(self):
        pass
    def train(self, fset, label, max_iter=30, th=1e-5,
              max_iter2=300,th2=1e-12):
        '''
        Train the MaxEnt model by a feature set fset
        Use Improved Iterative Scaling to solve the model
        :fset: a list of samples, each sample is a dict that contains
            the features
            e.g. [{'f1':0,'f2':1},{'f1':1,'f3':5}]
        :label: a list of labels. There must be len(fset)==len(label)
            e.g. ['l1','l2']
        '''
        # store the data in a sparce way so that
        # it is easier to retrieve a value
        enc, self.n_f, self.encf,self.labels=self.sparse_data(fset, label)
#        # make a aparce matrix of fset
#        fset2=({i:str(d[i]) for i in d} for d in fset)
#        v=DictVectorizer(np.bool)
#        X = v.fit_transform(fset2)
        # feature freq
        ff = np.zeros(self.n_f)
        for f, l in zip(fset,label):
            for i, v in self.encf(f, l):
                ff[i] += v
        ff=ff/len(fset) # normalize
        
        # map nf to integers from 0
        mapnf = set()
        for f in fset:
            for l in self.labels:
                mapnf.add(sum(v for (id,v) in self.encf(f, l)))
        mapnf = dict((nf, i) for (i, nf) in enumerate(mapnf))
        mapnp = np.array(sorted(mapnf, key=mapnf.get),np.float)
        mapnpt = mapnp.reshape((-1,1))
        ## use IIS
        lambdas=np.zeros(self.n_f)
        for _a in range(max_iter):
            print('iter:',_a)
            # solve the equation by Newton method
            deltas = np.ones(self.n_f)
            # compute a matrix for sum(p(x)p(y|x)f(x,y))
            pre = np.zeros((len(mapnf),self.n_f),np.float)
            for f in fset:
                for l in self.labels:
                    fvec=self.encf(f,l)
                    nf=np.sum(v for (id,v) in fvec) 
                    for id,v in fvec:
                        pre[mapnf[nf],id] += self.pred_prob(f,lambdas).prob(l) * v
            pre = pre/len(fset)
            # use Newton to solve the equation
            for _b in range(max_iter2):
                outer_mlp = np.outer(mapnp,deltas)
                exp_item = 2 ** outer_mlp
                t_exp = mapnpt * exp_item
                s1 = np.sum(exp_item * pre, axis=0)
                s2 = np.sum(t_exp * pre, axis=0)
                s2[s2==0]=1 #avoid 0
                dd = (ff-s1) / -s2
                deltas -= dd
                if np.abs(dd).sum()<th2 * np.abs(deltas).sum():
                    break
            lambdas += deltas
            self.lambdas=lambdas
            rate = np.abs(deltas).sum()/np.abs(lambdas).sum()
            print('R_lambda:',rate)
            if rate<th:
                break
        self.lambdas=lambdas
        return self
    
    def sparse_data(self,fset, label):
        # use the feature encoding class in nltk to store the sparse
        # matrix
        enc = sparse.store([n for n in zip(fset,label)])
        return enc,enc.length(),enc.encode,enc.labels()
        
    def pred_prob(self,f,w=None):
        'predict the probs of the labels'
        ps={}
        if w is None:
            w=self.lambdas
        for l in self.labels:
            fcode=self.encf(f,l)
            t=0.0
            for id,v in fcode:
                t+=w[id]*v
                ps[l]=t
        dist=DictionaryProbDist(ps,True,True)
        return dist
    
    def pred_class(self,f,w=None):
        ps = self.pred_prob(f,w)
        return ps.max()
    
    def test(self,fset):
        ans=[]
        for f in fset:
            ans.append(self.pred_class(f))
        return ans
    
    def save(self,fname='my_maxent2.model'):
        pd.io.pickle.to_pickle(self.__dict__,fname,'gzip')
    def load(self,fname='my_maxent2.model'):
        self.__dict__.update(pd.io.pickle.read_pickle(fname,'gzip'))
        return self

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
#    def save(self,fname='maxent_exc2.model'):
#        pd.io.pickle.to_pickle(self.__dict__,fname,'gzip')
#    def load(self,fname='maxent_exc2.model'):
#        self.__dict__.update(pd.io.pickle.read_pickle(fname,'gzip'))
#        return self

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
class multiclassify():
    def __init__(self):#,sumwds,excwds,tags):
        pass
        #self._get_ids(sumwds,excwds)
        #self.tags=tags
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
        

#%% The class for supporting the webpage
class webpage():
    def __init__(self):
        self.cl_sum=my_MaxEnt()
        self.cl_sum.load()
        self.cl_exc=pd.io.pickle.read_pickle('maxent_exc.model','gzip')
        self.tag_cl = multiclassify()
        self.tag_cl.load()
        self.tags = list(self.tag_cl.tags)
        self.wnl = WordNetLemmatizer()
        self.wdtoken = nltk.tokenize.WordPunctTokenizer().tokenize
        self.stopWords = set(stopwords.words('english'))
    def convert(self,w):
        'lemmatize trying different pos'
        #pos = ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
        for p in 'vnars':
            w1=self.wnl.lemmatize(w,pos=p)
            if w1!=w:
                break
        return w1
    def predict_fiction_prob(self,sum_f, exc_f):
        if exc_f is not None:
            return self.cl_exc.pipclassifier.prob_classify(exc_f).prob('Fiction')
        else:
            return self.cl_sum.pred_prob(sum_f).prob('Fiction')
    def predict_fiction(self,sum_f, exc_f):
        if self.predict_fiction_prob(sum_f, exc_f)>0.5:
            return 'Fiction'
        else:
            return 'Nonfiction'
    def predict_tag_prob(self,sum_f,exc_f,tag):
        if sum_f is None:
            return self.tag_cl.excmodel[tag].pipclassifier.prob_classify(exc_f).prob(True)
        if exc_f is None:
            return self.tag_cl.summodel[tag].pipclassifier.prob_classify(sum_f).prob(True)
        psum = self.tag_cl.summodel[tag].pipclassifier.prob_classify(sum_f).prob(True)
        pexc = self.tag_cl.excmodel[tag].pipclassifier.prob_classify(exc_f).prob(True)
        X = np.array([[psum,pexc]])
        return self.tag_cl.mixmodel[tag].predict_proba(X)[0,1] # 1 is prob(True)
    def predict_tags(self,sum_f,exc_f):
        ans = []
        for tag in self.tags:
            if self.predict_tag_prob(sum_f,exc_f,tag)>0.5:
                ans.append(tag)
        return sorted(ans)
    def preprocess(self,txt):
        tokens = [self.convert(w.lower()) for w in self.wdtoken(txt) if w.isalpha()]
        tokens = [w for w in tokens if w not in self.stopWords]
        return features(tokens) if tokens else None
    def predict_all(self,sum_txt,exc_txt):
        '''input 2 texts and output the results'''
        sum_f,exc_f = self.preprocess(sum_txt),self.preprocess(exc_txt)
        fictxt = self.predict_fiction(sum_f, exc_f)
        pred_tags = self.predict_tags(sum_f,exc_f)
        tagtxt = '\n'.join(pred_tags) if pred_tags else '(None)'
        return fictxt,tagtxt,len(pred_tags)
#%% test the functions
#testresult = pd.DataFrame(index=tag_cl.mix_testid,columns=tags)
#for id in tag_cl.mix_testid:
#    for tag in tags:
#        testresult.loc[id,tag] = predict_tag_prob(None, features(sumwds[id]),features(excwds[id]),tag)
#
#testresultTF = testresult>0.5
#trueresult = pd.DataFrame(False,index=tag_cl.mix_testid,columns=tags)
#for id in tag_cl.mix_testid:
#    for tag in tags:
#        if tag in genreslist[id]:
#            trueresult.loc[id,tag] = True
#
#correctresult = (testresultTF==trueresult)
#correctresult.mean()
#correctresult.mean().mean()
#((~correctresult).sum(axis=1)<=0).mean()
#0.22391857506361323
#((~correctresult).sum(axis=1)<=1).mean()
#0.638676844783715
#((~correctresult).sum(axis=1)<=2).mean()
#0.90585241730279897
#((~correctresult).sum(axis=1)<=3).mean()
#0.97455470737913485
#((~correctresult).sum(axis=1)<=4).mean()
#0.99491094147582693

#%% main function
def main():
    pass

if __name__=='__main__':
    main()