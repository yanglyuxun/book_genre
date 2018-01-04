#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN and Bidirectional LSTM for excerpt

"""

from read_data import read_text_data
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



#%% first steps
books,genre,excerpt = read_text_data()

MAX_WORDS = 10000
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(excerpt.values())
sequences = tokenizer.texts_to_sequences(excerpt.values())

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

lens = [len(s) for s in sequences]
sns.distplot(lens)
MAXLEN = 5000
data = pad_sequences(sequences, maxlen=MAXLEN)

#labels = to_categorical(np.asarray(labels))
labels = ['Fiction' in genre[id] for id in excerpt.keys()]
labels = np.array(labels).reshape((-1,1))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
ids = np.array(list(excerpt.keys()))
ids = ids[indices]
nb_validation_samples = int(0.3 * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
id_train = ids[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]
id_val = ids[-nb_validation_samples:]

#%% first training
#embedding_vecor_length = 32
#model = Sequential()
#model.add(Embedding(MAX_WORDS, embedding_vecor_length, input_length=MAXLEN))
#model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(LSTM(100))
#model.add(Dense(1, activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#print(model.summary())
#model.fit(x_train, y_train, epochs=3, batch_size=64)
##2917/2917 [==============================] - 229s - loss: 0.6526 - acc: 0.6394    
##Epoch 2/3
##2917/2917 [==============================] - 256s - loss: 0.5928 - acc: 0.6383     
##Epoch 3/3
##2917/2917 [==============================] - 269s - loss: 0.4216 - acc: 0.8080     
## Final evaluation of the model
#scores = model.evaluate(x_val, y_val, verbose=0)
#print("Accuracy: %f%%" % (scores[1]*100))
##80.304243


embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(MAX_WORDS, embedding_vecor_length, input_length=MAXLEN))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(100,dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

checkpoint = ModelCheckpoint("text_best_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

model.fit(x_train, y_train, epochs=3, batch_size=64,
          callbacks = [checkpoint, early])
# Final evaluation of the model
scores = model.evaluate(x_val, y_val, verbose=0)
print("Accuracy: %f%%" % (scores[1]*100))
#Epoch 1/3
#2917/2917 [==============================] - 471s - loss: 0.6559 - acc: 0.6356     
#Epoch 2/3
#2917/2917 [==============================] - 477s - loss: 0.5943 - acc: 0.6510     
#Epoch 3/3
#2917/2917 [==============================] - 471s - loss: 0.3844 - acc: 0.8348   
#Accuracy: 78.142514%  



model.save('text_model.h5')
pd.io.pickle.to_pickle([x_train,y_train,id_train,
                        x_val,y_val,id_val],'text_data.pc',
                        compression='gzip')



#%% continue training

model = load_model('text_model.h5')
x_train,y_train,id_train,x_val,y_val,id_val=pd.io.pickle.read_pickle('text_data.pc',
                        compression='gzip')
print(model.summary())

checkpoint = ModelCheckpoint("text_best_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')

model.fit(x_train, y_train, epochs=3, batch_size=64,
          callbacks = [checkpoint, early],
          validation_data=(x_val, y_val))
scores = model.evaluate(x_val, y_val, verbose=0)
print("Accuracy: %f%%" % (scores[1]*100))
