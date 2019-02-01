# -*- coding: utf-8 -*-
"""
Created on Sun May 27 00:31:45 2018

@author: Tejas_2

Testing results over GBM, RF, ExtraTR
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
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
# import xgboost as xgb

df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
df.drop(['society'], 1, inplace=True)
df.drop(['availability'],1, inplace=True)

# df = pd.get_dummies(data=df, columns=['availability', 'location'])
df = pd.get_dummies(data=df, columns=['location'])
df = df.dropna()

# Normalise all the numerical features. 
df['area_type'] = preprocessing.scale(df['area_type'])
df['total_sqft'] = preprocessing.scale(df['total_sqft'])
df['price'] = preprocessing.scale(df['price'])
df['size'] = preprocessing.scale(df['size'])
df['bath'] = preprocessing.scale(df['bath'])
df['balcony'] = preprocessing.scale(df['balcony'])


X = np.array(df.drop(['price'], 1))
# X = preprocessing.scale(X)
y = np.array(df['price'])
# y = 1 - np.log(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.10)

# clf = LinearRegression()
# GBM, n=70, depth=4, split=2, lR=0.5, max_features=20. - 73.6% accuracy
# max_leaf_nodes = k, then max_depth = k-1
clf1 = RandomForestRegressor(n_estimators=4000, n_jobs=-1)
                             #max_depth = 70, min_samples_split = 40, 
                              #   max_features=900, 
                               #  random_state=3, max_leaf_nodes=250, n_jobs=-1)

clf1.fit(X_train,y_train)

accuracy=clf1.score(X_test,y_test)
print(accuracy)

prediction = clf1.predict(X_test)
print(y_test, prediction)

score = mean_squared_error(prediction, y_test)
print(score)

clf2 = xgb.XGBRegressor(n_estimators=25, max_depth=4, colsample_bytree=0.6, 
                        min_child_weight=12, learning_rate=0.5, gamma=0.3,
                        max_delta_step=12, subsample=0.95, reg_alpha=0.7, reg_lambda=0.7
                        )
clf2.fit(X_train,y_train)

accuracy1=clf2.score(X_test,y_test)
print(accuracy1)

clf3 = GradientBoostingRegressor(n_estimators=25, max_depth = 15, min_samples_split = 130, 
                                 learning_rate = 0.1, max_features=90, subsample = 0.99, 
                                 random_state=3, max_leaf_nodes=70)

clf3.fit(X_train,y_train)

accuracy=clf3.score(X_test,y_test)
print(accuracy)