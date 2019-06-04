#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 02:17:31 2019

@author: chandra
"""

import pandas as pd
import numpy as np
import matplotlib as plt
#import pyplot 
import seaborn as sns

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
#dict_keys(['feature_names', 'target', 'data', 'DESR', 'target_name'])

#creating dataframe
df_feat= pd.DataFrame(cancer['data'], columns = cancer['feature_names'])
df_feat.head(2)

#data visualization skipped
 #train and fit data
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
X = df_feat
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=101)

from sklearn.svm import SVC
model=SVC()
model.fit(X_train, y_train)
prediction=model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, prediction))
print(classification_report(y_test, grid_prediction))



#Applying gridSearchCV
from sklearn.grid_search import GridSearchCV
param_grid = {'C':[0.1, 1, 10, 100, 1000], 'gamma':[1, 0.1, 0.01, 0.001, 0.0001]}
'''gamma[Hyperparameter]:
The hyperparameter γ controls the tradeoff between error due to bias and variance in your model.
If you have a very large value of gamma, then even if your two inputs are quite “similar”,
the value of the kernel function will be small - meaning that the support vector xn does not
have much influence on the classification of the training example xm.
This allows the SVM to capture more of the complexity and shape of the data,
but if the value of gamma is too large,
then the model can overfit and be prone to low bias/high variance.
On the other hand, a small value for gamma implies that the support vector has larger
influence on the classification of xm.
This means that the model is less prone to overfitting,
but you may risk not learning a decision boundary that captures the shape and complexity
of your data. This leads to a high bias, low variance model.
'''

grid = GridSearchCV(SVC(), param_grid, verbose=5,)
grid.fit(X_train, y_train)
grid.best_score_
grid.grid_scores_
grid.score
grid.best_estimator_
grid.best_params_
grid_predictions=grid.predict(X_test)
print(confusion_matrix(y_test, grid_predictions))
print(classification_report(y_test, grid_predictions))
sns.heatmap(confusion_matrix(y_test, grid_predictions), cmap='summer')





