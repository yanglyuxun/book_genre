#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 23:39:52 2017

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
#%% convert
#with open('booklist.pc','rb') as f:
#    booklist=pc.load(f)
#with open('sum_exc_gen.pc','rb') as f:
#    summarylist,excerptlist,genreslist=pc.load(f) 
#save([summarylist,excerptlist,genreslist], 'sum_exc_gen.pc')
#save(booklist,'booklist.pc')
    
# zip file
sumwds={}
excwds={}
with zipfile.ZipFile('txt_pro.zip') as zf:
    for id in genreslist:
        try:
            excpath='txt_pro/excerpt/'+str(id)+'.txt'
            excwds[id]=zf.read(excpath).decode("utf-8").split()
        except:
            pass
        try:
            sumpath='txt_pro/summary/'+str(id)+'.txt'
            sumwds[id]=zf.read(sumpath).decode("utf-8").split()
        except:
            pass

wnl = WordNetLemmatizer()
def convert(w):
    'lemmatize trying different pos'
    #pos = ADJ, ADJ_SAT, ADV, NOUN, VERB = 'a', 's', 'r', 'n', 'v'
    for p in 'vnars':
        w1=wnl.lemmatize(w,pos=p)
        if w1!=w:
            break
    return w1
for id in excwds:
    for i,w in enumerate(excwds[id]):
        excwds[id][i]=convert(w)
for id in sumwds:
    for i,w in enumerate(sumwds[id]):
        sumwds[id][i]=convert(w)

# save all to zip file
with zipfile.ZipFile('txt_pro_v2.zip',mode='w',compression=zipfile.ZIP_DEFLATED) as zf:
    for id in excwds:
        zf.writestr('excerpt/'+str(id)+'.txt',' '.join(excwds[id]))
    for id in sumwds:
        zf.writestr('summary/'+str(id)+'.txt',' '.join(sumwds[id]))

# save pickle
save([sumwds,excwds],'sum_exc_words.pc')

#%% updated part: output standard data for future uses
keep_col = ['author', 'title1', 'title2','url', 'imgurl']
newbl = booklist[keep_col]
newbl.to_csv('book_list.csv')

import json
excerpt = {str(id):t for id,t in excerptlist.items() if t}
with open('excerpt.json','w') as f:
    json.dump(excerpt,f)
summary = {str(id):t for id,t in summarylist.items() if t}
with open('summary.json','w') as f:
    json.dump(summary,f)
genre = {str(id):t for id,t in genreslist.items() if t}
with open('genre.json','w') as f:
    json.dump(genre,f)