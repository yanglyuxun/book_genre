#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
#%% how many fictions which have more than one tags?
fic_ids=[id for id in genreslist if 'Fiction' in genreslist[id] and len(genreslist[id])>1]
# 2974
fic_list=booklist.loc[fic_ids]


#%% how many cats?
genres=set()
for id in genreslist:
    for g in genreslist[id]:
        genres.add(g)
genres
for g in genres:
    print(g)

#%% frequency stat

def find_ids(exi=[],noexi=[]):
    ids=[id for id in genreslist]
    for g in exi:
        ids=[id for id in ids if g in genreslist[id]]
    for g in noexi:
        ids=[id for id in ids if g not in genreslist[id]]
    return ids
def find_counter(exi=[],noexi=[]):
    'count the tags where exi exists and noexi does not exist'
    ids=find_ids(exi,noexi)
    print(len(ids))
    gens=[]
    for id in ids:
        gens+=genreslist[id]
    return pd.DataFrame(Counter(gens).most_common())

temp=find_counter(['Fiction'])
temp=find_counter(['Nonfiction'])
temp=find_counter(["Children's Books"])
temp=find_counter(["Young Adult"])
temp=find_counter([],["Children's Books","Young Adult"])
both_id=find_ids(['Fiction','Nonfiction'])
temp=booklist.loc[both_id]

#%% imputation and correction!
# reorder the genres
for id in genreslist:
    if 'Fiction' in genreslist[id] and 'Nonfiction' in genreslist[id]:
        booklist.loc[id,'genre0']=np.nan
    elif 'Fiction' in genreslist[id]:
        booklist.loc[id,'genre0']='Fiction'
    elif 'Nonfiction' in genreslist[id]:
        booklist.loc[id,'genre0']='Nonfiction'
    else:
        booklist.loc[id,'genre0']=np.nan
    other=genreslist[id].copy()
    try:
        other.remove('Fiction')
    except:
        pass
    try:
        other.remove('Nonfiction')
    except:
        pass
    if len(other)>4: print('error!')
    for i in range(1,5):
        try:
            booklist.loc[id,'genre'+str(i)]=other[i-1]
        except:
            booklist.loc[id,'genre'+str(i)]=np.nan
booklist=booklist.drop('fiction',axis=1)

# imputation
gens=set()
for id in genreslist:
    gens.update(set(genreslist[id]))
gens.remove('Fiction')
gens.remove('Nonfiction')
gens={g:[0,0] for g in gens}
for id in genreslist:
    if 'Fiction' in genreslist[id]:
        for g in genreslist[id]:
            if g in gens:
                gens[g][0]+=1
    if 'Nonfiction' in genreslist[id]:
        for g in genreslist[id]:
            if g in gens:
                gens[g][1]+=1   
for g in gens:
    fic,nonfic=gens[g]
    ficrate=fic/(fic+nonfic)
    if ficrate>0.95:
        gens[g]='Fiction'
    elif ficrate<0.05:
        gens[g]='Nonfiction'
    else:
        gens[g]=np.nan
ficgens=[g for g in gens if gens[g]=='Fiction']
nficgens=[g for g in gens if gens[g]=='Nonfiction']
bothegens=[g for g in gens if gens[g] not in ['Fiction','Nonfiction']]

for id in booklist.index:
    isfic=0
    notfic=0
    for g in genreslist[id]:
        if g in ficgens:
            isfic+=1
        elif g in nficgens:
            notfic+=1
    if isfic>notfic:
        booklist.genre0[id]='Fiction'
    elif isfic<notfic:
        booklist.genre0[id]='Nonfiction'

# add text data and save as csv
summary=pd.DataFrame({'summary_text':summarylist})
excerpt=pd.DataFrame({'excerpt_text':excerptlist})
booklistall=booklist.join(summary).join(excerpt)

booklistall.to_csv('booklistall_v2.csv')

with open('booklist.pc','wb') as f:
    pc.dump(booklistall,f)

#%% stat for chars
text_col=['author', 'title1', 'title2','summary_text', 'excerpt_text']
textlist=booklist[text_col]
txtall=''
for c in textlist:
    for t in textlist[c]:
        txtall+=t
temp=Counter(txtall).most_common()
known=' \nQWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890'
temp2=[i for i in temp if i[0] not in known]
temp2=pd.DataFrame(temp2)

text_col=['author', 'title1', 'title2','summary_text', 'excerpt_text']
booklist3=booklist.copy()
needed="-–—'’QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890"
letters="QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm"
for c in text_col:
    for i in booklist3.index:
        oldstr=booklist3.loc[i,c]
        oldstr='\n'.join([i.strip() for i in oldstr.split('\n')])
        oldstr=oldstr.replace('-\n','')
        oldstr=oldstr.replace('\n-','')
        oldstr=oldstr.replace('-\n-','')
        oldstr=oldstr.replace('–\n','')
        oldstr=oldstr.replace('\n–','')
        oldstr=oldstr.replace('–\n–','')
        oldstr=oldstr.replace('—\n','')
        oldstr=oldstr.replace('\n—','')
        oldstr=oldstr.replace('—\n—','')
        newstr=''
        for p,ch in enumerate(oldstr):
            if ch in "'’":
                try:
                    if (oldstr[p-1] in letters) and (oldstr[p+1] in letters):
                        newstr+=ch
                    else:
                        newstr+=' '
                except:
                    newstr+=' '
            elif ch in needed:
                newstr+=ch
            else:
                newstr+=' '
        booklist3.loc[i,c]=newstr
booklist3.to_csv('booklistall_v3.csv')
with open('booklist3.pc','wb') as f:
    pc.dump(booklist3,f)

#%% save summary and excerpt to txt
for id in summarylist:
    if summarylist[id]:
        with open('txt_summary/'+str(id)+'.txt','w') as f:
            f.write(summarylist[id])
    if excerptlist[id]:
        with open('txt_excerpt/'+str(id)+'.txt','w') as f:
            f.write(excerptlist[id])

