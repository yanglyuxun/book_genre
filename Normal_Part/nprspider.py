#-*-coding:utf-8-*-

import requests
from bs4 import BeautifulSoup
import pickle as pc
import logging
import time
import pandas as pd
import numpy as np


headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36',
         'Referer':'http://www.npr.org/books/'}

#%%
rooturl='http://www.npr.org/books/'

rootpg=requests.get(rooturl, headers=headers)
rtsoup = BeautifulSoup(rootpg.content)
allas = [str(a0) for a0 in [a.get('href') for a in rtsoup.find_all('a')] if str(a0).startswith('/books/genres/')]
genres0 = {a.split('/')[-2]:'http://www.npr.org'+a for a in allas}
needed_genres = ['biography-memoir',
 'children',
 'comedy',
 'comics-graphic-novels',
 'digital-culture',
 'faith-spirituality',
 'food-wine',
 'history-society',
 'historical-fiction',
 'horror-supernatural',
 'literary-fiction',
 'mystery-thrillers',
 'parenting-families',
 'politics-public-affairs',
 'romance',
 'science-fiction-fantasy',
 'science-health',
 'sports',
 'travel',
 'young-adults']
genres = {i:genres0[i] for i in needed_genres}
with open('genres_urls.pc','wb') as f:
    pc.dump(genres,f)

# get all the <article>s
for g in genres:
    achurl=genres[g]+'archive/'
    dateend='12-31-2015'
    start=0
    pages=[]
    logging.basicConfig(filename=g+'_log.log',level=logging.DEBUG)
    while True:
        time.sleep(1)
        nowurl=achurl+r'?start='+str(start)+r'&date=12-31-2015'
        pg=requests.get(nowurl, headers=headers)
        pages.append(pg)
        with open(g+'_pages.pc','wb') as f:
            pc.dump(pages,f)
        soup = BeautifulSoup(pg.content,"lxml")
        arts = soup.find_all('article')
        logging.debug('Done with '+nowurl+', find arts: '+str(len(arts)))
        if len(arts):
            logging.debug('First title: '+arts[0].a.get('href').split('/')[-1])
        else:
            logging.debug('End of this genre!')
            break
        start+=15
        if start>=1500:
            break

# stat for the article lists
#stat={}
#all_art={}
#for g in genres:
#    with open(g+'_pages.pc','rb') as f:
#        pages=pc.load(f)
#    art_dict={}
#    for pg in pages:
#        soup = BeautifulSoup(pg.content,"lxml")
#        arts = soup.find_all('article')
#        art_dict.update({art.a.get('href').split('/')[-2]:art for art in arts})
#    all_art.update(art_dict)
#    stat[g]=len(art_dict)
#stat
#len(all_art)
#with open('stat.pc','wb') as f:
#    pc.dump([stat,len(all_art)],f)

#%%
# build the whole list
# booklist=pd.DataFrame(columns=['id','name','author,'origin','index_in_origen','url'])
booklist={}
with open('genres_urls.pc','rb') as f:
    genres=pc.load(f)
for g in genres:
    with open(g+'_pages.pc','rb') as f:
        pages=pc.load(f)
    index_in_origen=0
    for pg in pages:
        soup = BeautifulSoup(pg.content,"lxml")
        arts = soup.find_all('article')
        for art in arts:
            book={}
            book['name']=art.h2.string
            book['url']=art.h2.a.get('href')
            try:
                book['author']=art.find_all('p','author')[0].a.string
            except:
                try:
                    book['author']=art.find_all('p','author')[0].get_text().replace('by ','')
                except:
                    print(art.find_all('p','author'))
                    book['author']=''
            book['index_in_origen']=index_in_origen
            index_in_origen+=1
            book['origin']=g
            booklist[int(book['url'].split('/')[-2])]=book
    print(g,'Done')
for id in booklist:
    for nm in booklist[id]:
        booklist[id][nm]=str(booklist[id][nm])
booklist=pd.DataFrame(booklist).transpose()
with open('booklist.pc','wb') as f:
    pc.dump(booklist,f)
    
#%% get the pages of all books (shuffle in case cannot get all of them)

with open('booklist.pc','rb') as f:
    booklist=pc.load(f)
booklist['downloaded']=False
no=0
allpages={}
logging.basicConfig(filename='allpage_log.log',level=logging.DEBUG)
for id in booklist.index:
    pg=requests.get(booklist.loc[id,'url'], headers=headers)
    allpages[id]=pg.content
    booklist.loc[id,'downloaded']=True
    logging.debug('ID: '+str(id))
    print(id,'check')
    if len(allpages)%300==0:
        print(len(allpages))
        with open('allpages'+str(no)+'.pc','wb') as f:
            pc.dump(allpages,f)
        allpages={}
        no+=1
        time.sleep(5)
    time.sleep(1)
with open('allpages'+str(no)+'.pc','wb') as f:
    pc.dump(allpages,f)
logging.debug('FINISHED!')


#%% orgnize all the data
with open('booklist.pc','rb') as f:
    booklist=pc.load(f)
allpages={}
for i in range(24):
    with open('./allpages/allpages'+str(i)+'.pc','rb') as f:
        allpages.update(pc.load(f))

booklist['summary']=False
booklist['excerpt']=False
booklist['title2']=''
booklist['imgurl']=''
booklist['img']=False
summarylist={}
excerptlist={}
genreslist={}
for id in allpages:
    pg=allpages[id]
    soup = BeautifulSoup(pg,"lxml")
    if soup.find('h1').text != booklist.name[id]:
        print(id,'name not same: ',soup.find('h1').text,booklist.name[id])
        Exception()
    try:
        booklist.loc[id,'title2']=soup.find('div',{'class':'booktitle'}).find('h2').text
    except:
        pass
    try:
        summary=soup.find('div',{'id':'summary'}).text.strip()
        summarylist[id]=summary
        booklist.loc[id,'summary']=True
    except:
        summarylist[id]=''
    genres=soup.find('div',{'id':'bookmeta'}).find_all('a')
    genres=[a.text for a in genres]
    genreslist[id]=genres
    try:
        excerpt=soup.find('div',{'id':'storytext'}).find_all('p')
        excerptlist[id]='\n'.join([a.text for a in excerpt]).strip()
        booklist.loc[id,'excerpt']=True
    except:
        excerptlist[id]=''
    try:
        booklist.loc[id,'imgurl']=soup.find('img',{'class':'img'}).get('src')
        booklist.loc[id,'img']=True
    except:
        pass
    if len(summarylist)%100==0: # show the pregress
        print(len(summarylist)/len(allpages))
with open('sum_exc_gen.pc','wb') as f:
    pc.dump([summarylist,excerptlist,genreslist],f) 

# add genres to booklist
maxgen=0
for id in genreslist:
    if len(genreslist[id])>maxgen:
        maxgen=len(genreslist[id])
print(maxgen)
for i in range(maxgen):
    booklist['genre'+str(i)]=''
for id in genreslist:
    for i,t in enumerate(genreslist[id]):
        booklist.loc[id,'genre'+str(i)]=t

booklist['fiction']=np.nan
for id in genreslist:
    if 'Fiction' in genreslist[id]:
        booklist.loc[id,'fiction']=True
    elif  'Nonfiction' in genreslist[id]:
        booklist.loc[id,'fiction']=False

with open('booklist.pc','wb') as f:
    pc.dump(booklist,f)

#%% get all images
# make dir img/
with open('booklist.pc','rb') as f:
    booklist=pc.load(f)
for id,url in booklist.imgurl.iteritems():
    if url.find('-s99-c15')==0: print('not found')
N,n=booklist.img.sum(),0
for id,url in booklist.imgurl.iteritems():
    if booklist.loc[id,'img']==False:
        continue
    url=url.replace('-s99-c15','')
    im=requests.get(url, headers=headers)
    with open('./img/'+str(id)+'.jpg','wb') as f:
        f.write(im.content)
    n+=1
    if n%100==0:
        print(n/N)

#%%######################!!!!!!
#here use nprspider2.py and nprspider4.py to add data.
#########################!!!!!!
#%% output to csv
with open('booklist.pc','rb') as f:
    booklist=pc.load(f)
with open('sum_exc_gen.pc','rb') as f:
    summarylist,excerptlist,genreslist=pc.load(f) 
    
booklist['title1']=booklist.pop('name') # change name to title1!
newcols=['author','title1','title2','genre0', 'genre1', 'genre2', 'genre3',
       'genre4','fiction','excerpt','summary','img','origin','index_in_origen',  
       'url', 'imgurl']
booklist=booklist[newcols]
with open('booklist.pc','wb') as f:
    pc.dump(booklist,f)
#booklist.to_csv('booklist.csv')

summary=pd.DataFrame({'summary_text':summarylist})
excerpt=pd.DataFrame({'excerpt_text':excerptlist})
booklistall=booklist.join(summary).join(excerpt)

booklistall.to_csv('booklistall.csv')
