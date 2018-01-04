#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image
"""

from skimage.io import imread
import pandas as pd
import numpy as np
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import sklearn

import gzip
import pickle
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


#%% read data

imgs = {}
_n=0
for id in booklist.index:
    try:
        img = imread('img/'+str(id)+'.jpg',as_grey=False)
        imgs[id] = resize(img,(20,20),mode='reflect')        
    except:
        pass
    _n+=1
    if _n%100==0:
        print(_n/9800)
        
idsimg = [i for i in imgs if not pd.isnull(booklist.genre0.loc[i])]
trainX=[]
testX=[]
trainy=[]
testy=[]
import random
random.seed(777)
for id in idsimg:
    if id in testid:
        testX.append(imgs[id].reshape((1,-1)))
        testy.append(booklist.genre0.loc[id])
    elif id in trainid:
        trainX.append(imgs[id].reshape((1,-1)))
        trainy.append(booklist.genre0.loc[id])
    else:
        if random.random()>0.1:
            trainX.append(imgs[id].reshape((1,-1)))
            trainy.append(booklist.genre0.loc[id])      
        else:
            testX.append(imgs[id].reshape((1,-1)))
            testy.append(booklist.genre0.loc[id])
trainX = np.concatenate(trainX,axis=0)
testX = np.concatenate(testX,axis=0)

save([trainX,testX,trainy,testy],'img_dataset.pc')

cl=sklearn.svm.SVC(verbose=True)
cl.fit(trainX,trainy)
cl.score(trainX,trainy)
0.63826086956521744
cl.score(testX,testy)
0.64984227129337535
