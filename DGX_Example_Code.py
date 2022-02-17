#!/usr/bin/env python
# coding: utf-8

# In[13]:


import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import json


# ## Load Data

# In[2]:


train = pd.read_csv('fashion_mnist/fashion-mnist_train.csv')
test = pd.read_csv('fashion_mnist/fashion-mnist_test.csv')


# In[3]:


train.shape


# In[4]:


test.shape


# In[5]:


X_train = train.drop(['label'], axis=1).to_numpy()
Y_train = train['label'].to_numpy()

X_test = test.drop(['label'], axis=1).to_numpy()
Y_test = test['label'].to_numpy()


# In[6]:


X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255


# In[7]:


class_names = ['T_shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# # Model

# In[8]:


model = Sequential([
    Conv2D(32,(3,3),input_shape=(28,28,1), activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64,(3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.25),
    Dense(10, activation='softmax')
])


# In[9]:


model.compile(loss="sparse_categorical_crossentropy",optimizer='adam', metrics=['accuracy'])


# In[10]:


model.summary()


# In[11]:


hist = model.fit(X_train,Y_train, validation_data=(X_test,Y_test), epochs=10, verbose=2, workers=4)


# In[12]:


model.save('my_model')


# In[14]:


with open('log.json', 'w') as file:
    json.dump(hist.history,file)


# In[ ]:




