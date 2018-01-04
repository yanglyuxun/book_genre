
# coding: utf-8

# In[20]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, CSVLogger
import pandas as pd
import numpy as np
import zipfile,os
from skimage.io import imread
from skimage.transform import rescale, resize, downscale_local_mean
import read_data 

img_width, img_height = 224, 224
train_data_dir = "data/img_train"
validation_data_dir = "data/img_val"
nb_train_samples = 4125
nb_validation_samples = 466 
batch_size = 16
epochs = 50

from pandas.io.pickle import to_pickle, read_pickle
save_dir = 'img_classifier1/'
import os
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# In[5]:


#%% make data
_,genre,_ = read_data.read_text_data()
id_train,id_val = read_data.read_ids()


# In[6]:


if not os.path.exists(train_data_dir):
    os.mkdir(train_data_dir)
    os.mkdir(validation_data_dir)
    os.mkdir(train_data_dir+'/fic')
    os.mkdir(train_data_dir+'/nonfic')
    os.mkdir(validation_data_dir+'/fic')
    os.mkdir(validation_data_dir+'/nonfic')
    with zipfile.ZipFile('data/NPR_data_covers.zip') as zf:
        fnames = [f.filename for f in zf.infolist()]
        fnames.pop(0)
        ids = [f.replace('img/','').replace('.jpg','') for f in fnames]
        for id in ids:
            g = 'fic' if 'Fiction' in genre[id] else 'nonfic'
            with zf.open('img/%s.jpg'%id) as f:
                dir1 = train_data_dir if id in id_train else validation_data_dir
                with open(dir1+'/%s/%s.jpg'%(g,id),'wb') as fout:
                    fout.write(f.read())


# In[24]:


#%% import model
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))

# Freeze the layers 
for layer in model.layers:
    layer.trainable = False

model.summary()


# In[32]:


#Adding custom Layers 
x = model.output
x = Flatten()(x)
#x = Dense(128, activation="relu")(x)
#x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
#predictions = Dense(16, activation="softmax")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(inputs = [model.input], outputs = [predictions])

# compile the model 
#model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])
model_final.compile(loss = "categorical_crossentropy", optimizer = 'adam', metrics=["accuracy"])
model_final.summary()


# In[33]:


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


# In[34]:


# Save the model according to the conditions  
checkpoint = ModelCheckpoint(save_dir+"img_best_model.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
with open(save_dir+'text_csvlogger.csv','w') as f:
    f.write('')
csvlog = CSVLogger(save_dir+'text_csvlogger.csv',append=True)


# In[35]:


# Train the model 
model_final.fit_generator(
    train_generator,
    steps_per_epoch = int(nb_train_samples/batch_size),
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = int(nb_validation_samples/batch_size),
    callbacks = [checkpoint, csvlog])
model_final.save(save_dir+'img_model_1.h5')

