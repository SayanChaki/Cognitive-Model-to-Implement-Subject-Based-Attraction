# -*- coding: utf-8 -*-
"""
Created on Tue May 18 21:53:42 2021

@author: DELL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


data=pd.read_csv('Attraction3_training.csv')
data.head()
data.info()
data['Attracted'],_=pd.factorize(data['Attracted'])
X=data.iloc[:,12:21]
Y=data.iloc[:,21]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20)
svclassifier = SVC(kernel='sigmoid')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
