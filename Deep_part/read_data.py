#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
read crawled data from NPR
make a train/val split on all available IDs

@author: ylx
"""

import zipfile, json
import pandas as pd
import numpy as np

#with zipfile.ZipFile('data/NPR_data_text.zip') as zf:
#    for f in zf.infolist():
#        print(f.filename)
#with zipfile.ZipFile('data/NPR_data_covers.zip') as zf:
#    print(len(zf.infolist()))
#    for f in zf.infolist()[:10]:
#        print(f.filename)

def read_text_data():
    with zipfile.ZipFile('data/NPR_data_text.zip') as zf:
        with zf.open('excerpt.json','r') as f:
            excerpt = json.loads(f.read().decode())
    #    with zf.open('summary.json') as f:
    #        summary = json.load(f)
        with zf.open('genre.json','r') as f:
            genre = json.loads(f.read().decode())
        with zf.open('book_list.csv','r') as f:
            books = pd.read_csv(f,index_col=0)
    return books,genre,excerpt


def split_ids(seed=777, val_prop=0.3):
    books,_,_ = read_text_data()
    ids = [str(id) for id in books.index]
    np.random.seed(seed)
    np.random.shuffle(ids)
    n_val = round(val_prop*len(ids))
    id_train = ids[:-n_val]
    id_val = ids[-n_val:]
    return id_train,id_val

def read_ids():
    with open('data/id_split.json','r') as f:
        dic = json.load(f)
    return dic['id_train'],dic['id_val']

#%% save the id split results
if __name__=='__main__':
    with open('data/id_split.json','w') as f:
        id_train,id_val = split_ids()
        json.dump({'id_train':id_train,
                   'id_val':id_val},f)
