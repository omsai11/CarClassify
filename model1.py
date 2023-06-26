#!/usr/bin/env python
# coding: utf-8

# In[3]:


#import tensorflow as tf
#from tensorflow import keras
import keras
from keras.optimizers import Nadam
from keras.preprocessing.image import ImageDataGenerator
import scipy


# In[4]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(7, activation='softmax')])
model.compile(loss='categorical_crossentropy',optimizer='Nadam',metrics=['accuracy'])
    # All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1./255,)

    # Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    'train',  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 300x300
    batch_size=128,
    class_mode='categorical')

model.fit(
    train_generator,
    steps_per_epoch=8,  
    epochs=15,)


# In[ ]:


def predictfunc(x):
    predictions=model.predict(x)
    return predictions

