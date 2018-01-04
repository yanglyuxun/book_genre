#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MaxEnt
"""
import numpy as np
import pandas as pd
import math
#from sklearn.feature_extraction import DictVectorizer
from sparse_class import sparse
from functools import reduce
from nltk.probability import DictionaryProbDist

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
    
    def save(self,fname='my_maxent.model'):
        pd.io.pickle.to_pickle(self,fname,'gzip')
    def load(self,fname='my_maxent.model'):
        self = pd.io.pickle.read_pickle(fname,'gzip')
        return self

#%% train my model
# (data are from the previous file '2_fic_classifier.py')
fset, label = zip(*sum_train)
cla=my_MaxEnt()
cla.train(fset,label)
testfset, testlabel = zip(*sum_test)
pred_label=cla.test(testfset)
np.mean(np.array(pred_label)==np.array(testlabel)) #accuracy
# 0.61520190023752974
cla.save()

#%% load two models and train weights
# (load the data from the previous file '2_fic_classifier.py')
cl_sum=my_MaxEnt()
cl_sum=cl_sum.load()
cl_exc=pd.io.pickle.read_pickle('maxent_exc.model','gzip')

# make data
# summary
sum_featuresets = {c:(features(d), booklist.genre0.loc[c]) for c,d in sumwds.items()
        if c in ids}
# excerpt
exc_featuresets = {c:(features(d), booklist.genre0.loc[c]) for c,d in excwds.items()
        if c in ids}
# data table
data=pd.DataFrame(index=ids,columns=['sum_fic','exc_fic','label'])
for id in ids:
    data.loc[id,'sum_fic']=cl_sum.pred_prob(sum_featuresets[id][0]).prob('Fiction')
    data.loc[id,'exc_fic']=cl_exc.pipclassifier.prob_classify(exc_featuresets[id][0]).prob('Fiction')
    data.loc[id,'label']=booklist.genre0.loc[id]
mix_train=data.loc[trainid]
mix_test=data.loc[testid]
X_train=mix_train.iloc[:,0:2]
X_test=mix_test.iloc[:,0:2]
y_train=list(mix_train.iloc[:,2].values)
y_test=list(mix_test.iloc[:,2].values)
((X_test.iloc[:,1].as_matrix()>0.5) == (np.array(y_test)=='Fiction')).mean()
0.90025575447570327
((X_test.iloc[:,0].as_matrix()>0.5) == (np.array(y_test)=='Fiction')).mean()
0.64450127877237851

# train the mix part
mix_model = sklearn.linear_model.LogisticRegression(class_weight='balanced')
mix_model.fit(X_train,y_train)
np.mean(mix_model.predict(X_train)==y_train)
np.mean(mix_model.predict(X_test)==y_test)
mix_model.score(X_test,y_test)
0.76214833759590794

plt.figure()

w=np.array(range(1,1000))/1000
acc=lambda x:((np.sum(X_train.as_matrix()* np.array([x,1-x]),axis=1)>0.5)==(np.array(y_train)=='Fiction')).mean()
plt.plot(w,list(map(acc,w)),label='Train')

acc=lambda x:((np.sum(X_test.as_matrix()* np.array([x,1-x]),axis=1)>0.5)==(np.array(y_test)=='Fiction')).mean()
plt.plot(w,list(map(acc,w)),label='Test')

plt.xlabel('weight')
plt.ylabel('accuracy')
plt.legend()
plt.title('Combination of two models')
