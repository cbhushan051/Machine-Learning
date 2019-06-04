#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 17:49:37 2019

@author: chandra
"""
#imports
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns

from sklearn import datasets
from sklearn.multiclass import OutputCodeClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing 
from scipy.io import loadmat

#getdataset
data = loadmat('matlab_fc6.mat')
data.keys()
testdata = data['featuresTest']
traindata = data['featuresTrain']

a=traindata[14550:15550,:]#Taking Benign and Malignant(500+500)
X_train=a 
X_test=testdata
'''
#Taking whole data for training and testing
X_train=traindata
X_test=testdata

'''
#Normalizing Data
X_test = preprocessing.normalize(X_test)
X_train = preprocessing.normalize(X_train)

#labelling y_train
x= np.array(np.zeros(500), ndmin=1)    #label 0 for benign
y= np.array(np.ones(500), ndmin=1)     #label 1 for malignant
y_train=np.concatenate((x,y), axis=0)
'''
x= np.array(np.zeros(15050), ndmin=1)    #label 0 for benign
y= np.array(np.ones(15050), ndmin=1)     #label 1 for malignant
y_train=np.concatenate((x,y), axis=0)

'''

#labeling y_test
x1= np.array(np.zeros(50), ndmin=1)  #label 0 for benign
y1= np.array(np.ones(50), ndmin=1)   #label 1 for malignant
y_test=np.concatenate((x1,y1), axis=0)

################Using LinearSVC
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000,5000, 10000]}
clf = OutputCodeClassifier(LinearSVC(random_state=0, verbose=5),
                           code_size=3, random_state=0)
clf.fit(X_train, y_train)
predictions=clf.predict(X_test)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

###########Using GridSearchCV
param_grid = {'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000,5000, 10000], 'gamma':[100,10,1,0.1,0.01,0.001,0.0001]}
model_grid = GridSearchCV(SVC(), param_grid, verbose=5,cv=10)


ecoc = OutputCodeClassifier(LinearSVC(random_state=0), random_state=0)
Cs = [0.0001,0.001, 0.01,0.5, 0.8, 0.1, 1, 10, 100, 1000, 5000, 10000]
cv = GridSearchCV(ecoc, {'estimator__C': Cs}, verbose=5, cv=10)

cv.fit(X_train,y_train)
grid_pred = cv.predict(X_test)
cv.best_params_

print(confusion_matrix(y_test, grid_pred))
print(classification_report(y_test, grid_pred))

#Visualizing
sns.heatmap(confusion_matrix(y_test, grid_pred), cmap='summer')

############Using SVC
from sklearn.svm import SVC
C_range = np.logspace(-2, 10, 13) 
gamma_range = np.logspace(-9, 3, 13)
param_grid = dict(gamma=gamma_range, C=C_range)
'''
Trying to provide 2D target for binary classifier on two classes but failed!!!
#labeling y_train
c=np.array(np.zeros(1000) , ndmin=1)
c[0:500,]=1
a=np.array(np.zeros(1000) , ndmin=1)
a[500:1000,]=1

y_train=np.array((c,a))
y_train=np.transpose(y_train)

#labeling y_test
c=np.array(np.zeros(50) , ndmin=1)
c[0:50,]=1
a=np.array(np.zeros(50) , ndmin=1)
a[50:100,]=1

y_train=np.array((c,a))
y_train=np.transpose(y_train)
'''






ecoc = OutputCodeClassifier(SVC(random_state=0), random_state=0)
Cs = [0.0001,0.001, 0.01,0.5, 0.8, 0.1, 1, 10, 100, 1000, 5000, 10000]
cv = GridSearchCV(ecoc, {'estimator__C': Cs}, verbose=5, cv=10)

ecoc.fit(X_train, y_train)

cv.fit(X_train,y_train)
grid_pred = cv.predict(X_test)
cv.best_params_

print(confusion_matrix(y_test, grid_pred))
print(classification_report(y_test, grid_pred))

#Visualizing
sns.heatmap(confusion_matrix(y_test, grid_pred), cmap='summer')


