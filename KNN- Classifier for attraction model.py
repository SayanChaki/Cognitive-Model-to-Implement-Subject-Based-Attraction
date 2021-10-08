# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:45:03 2021

@author: DELL
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

data=pd.read_csv('Attraction3_training.csv')
data.head()
data.info()
data['Attracted'],_=pd.factorize(data['Attracted'])
x=data.iloc[:,12:21]
y=data.iloc[:,21]
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)
print(x_train)
print(x_test)
print(x_test)
KNN_model=KNeighborsClassifier(n_neighbors=9,p=2,metric='euclidean')
KNN_model.fit(x_train,y_train)
y_pred=KNN_model.predict(x_test)
print(y_pred)
print([[1]])
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(f1_score(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
