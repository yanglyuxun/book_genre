#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image classifier

"""

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
import pandas as pd
import numpy as np
import zipfile,os
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
from read_data import read_text_data

img_width, img_height = 256, 256
train_data_dir = "data/img_train"
validation_data_dir = "data/img_val"
nb_train_samples = 4125
nb_validation_samples = 466 
batch_size = 16
epochs = 50

#%% make data
books,genre,excerpt = read_text_data()
x_train,y_train,id_train,x_val,y_val,id_val=pd.io.pickle.read_pickle('text_data.pc',
                        compression='gzip')
with zipfile.ZipFile('data/NPR_data_covers.zip') as zf:
    fnames = [f.filename for f in zf.infolist()]
fnames.pop(0)
ids = [f.replace('img/','').replace('.jpg','') for f in fnames]
id_train_img = []
id_val_img = []
ids0 = []
for id in ids:
    if id in id_train:
        id_train_img.append(id)
    elif id in id_val:
        id_val_img.append(id)
    else:
        ids0.append(id)
validation_samples = int(0.3 * len(ids0))
np.random.shuffle(ids0)
id_train_new = ids0[:-validation_samples]
id_val_new = ids0[-validation_samples:]
id_train_img += id_train_new
id_val_img += id_val_new

os.mkdir(train_data_dir)
os.mkdir(validation_data_dir)
os.mkdir(train_data_dir+'/fic')
os.mkdir(train_data_dir+'/nonfic')
os.mkdir(validation_data_dir+'/fic')
os.mkdir(validation_data_dir+'/nonfic')
with zipfile.ZipFile('data/NPR_data_covers.zip') as zf:
    for id in id_train_img:
        with zf.open('img/%s.jpg'%id) as f:
            if 'Fiction' in genre[id]:
                g = 'fic'
            else:
                g = 'nonfic'
            with open(train_data_dir+'/%s/%s.jpg'%(g,id),'wb') as fout:
                fout.write(f.read())
    for id in id_val_img:
        with zf.open('img/%s.jpg'%id) as f:
            if 'Fiction' in genre[id]:
                g = 'fic'
            else:
                g = 'nonfic'
            with open(validation_data_dir+'/%s/%s.jpg'%(g,id),'wb') as fout:
                fout.write(f.read())


#%% build model
model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

model.summary()

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:17]:
    layer.trainable = False

#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
#predictions = Dense(16, activation="softmax")(x)
predictions = Dense(1, activation="sigmoid")(x)

# creating the final model 
model_final = Model(input = model.input, output = predictions)

# compile the model 
#model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.compile(loss = "binary_crossentropy", optimizer = 'adam', metrics=["accuracy"])
# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = "nearest",
    zoom_range = 0.1)

test_datagen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range=0.1,
    height_shift_range=0.1,
    fill_mode = "nearest",
    zoom_range = 0.1)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical")

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# Train the model 
model_final.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
    epochs = epochs,
    validation_data = validation_generator,
    nb_val_samples = nb_validation_samples,
    callbacks = [checkpoint, early])