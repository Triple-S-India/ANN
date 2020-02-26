#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Artificial Neural Network


# In[2]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[3]:


#importing the dataset
dataset = pd.read_csv("D:DS_TriS/Churn_Modelling.csv")
dataset.head()


# In[4]:


dataset.shape


# In[5]:


X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values


# In[6]:


X[0:5], y[0:5]


# In[7]:


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
X[0:5]


# In[8]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[9]:


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[10]:



# Part 2 - Now let's make the ANN!


# In[11]:


# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[12]:


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


# In[13]:


# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[14]:


# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


# In[15]:



# Part 3 - Making the predictions and evaluating the model


# In[16]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# In[17]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[18]:


cm


# In[21]:


accuracy = (1555+128)/2000
accuracy


# In[ ]:




