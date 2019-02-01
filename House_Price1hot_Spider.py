# -*- coding: utf-8 -*-
"""
Created on Sun May 27 00:31:45 2018

@author: Tejas_2
"""

# Import all stuff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, svm, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
df.drop(['society'], 1, inplace=True)
df.drop(['availability'],1, inplace=True)

# df = pd.get_dummies(data=df, columns=['availability', 'location'])
df = pd.get_dummies(data=df, columns=['location'])
df = df.dropna()

# Normalise all the numerical features. 
# df['area_type'] = preprocessing.scale(df['area_type'])
# df['total_sqft'] = preprocessing.scale(df['total_sqft'])
# df['price'] = preprocessing.scale(df['price'])



X = np.array(df.drop(['price'], 1))
# X = preprocessing.scale(X)
y = np.array(df['price'])
y = 1 - np.log(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)

# clf = LinearRegression()
# GBM, n=70, depth=4, split=2, lR=0.5, max_features=20. - 73.6% accuracy
# 250, 15, 30, 0.1, max_features=150, subsample=0.94, random_state=5, max_leaf_nodes=41 = 84.4%
# max_leaf_nodes = k, then max_depth = k-1
# if n estimatorsis increased, decrease learning rate too.
clf1 = GradientBoostingRegressor(n_estimators=500, max_depth = 15, min_samples_split = 30, 
                                 learning_rate = 0.1, max_features=150, subsample = 0.94, 
                                 random_state=5, max_leaf_nodes=41)

clf1.fit(X_train,y_train)

accuracy=clf1.score(X_test,y_test)
print(accuracy)

prediction = clf1.predict(X_test)
#print(y_test, prediction)
actual_prediction = np.exp(1-prediction)
actual_y_test = np.exp(1-y_test)
score = np.sqrt(mean_squared_error(np.log(actual_prediction), np.log(actual_y_test)))
print(1-score)
