# -*- coding: utf-8 -*-
"""
Created on Thu May 31 01:16:57 2018

@author: Tejas_2

Title: House Price prediction, 3rd submisssion- 87.5% accuracy, 3rd in Hackathon
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, svm, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor

# Input the test and Train datasets
df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
df.drop(['society'], 1, inplace=True)
df.drop(['availability'],1, inplace=True)
#df = df.fillna(0)

df1 = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Test-Data.csv')
df1.drop(['society', 'availability', 'price'], 1, inplace=True)
#df1 = df1.fillna(0)

df = pd.get_dummies(data=df, columns=['location'])
df1 = pd.get_dummies(data=df1, columns=['location'])

# Once encoded drop the unwantedly, manually added columns
# The manual addition was just to get correct features during one hot encoding.
df = df.dropna(axis=0)
df1 = df1.dropna(axis=0)
# Convert them to computable arrays 
X = np.array(df.drop(['price'], 1))
y = np.array(df['price'])
y = 1 - np.log(y)
#y = np.log(y)

X_test = np.array(df1)

# Train the classifier, With this accuracy of 84.277% was achieved.
# n=250, max_depth=15, min_samples_split=30, learning_rate=0.1, max_features =150,
# subsample 0.94, max_leaf_nodes=41.
clf1 = GradientBoostingRegressor(n_estimators=500, max_depth = 15, min_samples_split = 30,
                                 learning_rate = 0.1, max_features=150, subsample = 0.94, 
                                 random_state=3, max_leaf_nodes=41)

# Fit the trained model to the dataset
clf1.fit(X,y)

# Now predict the price values in the test dataset.
prediction = clf1.predict(X_test)
actual_prediction = np.exp(1-prediction)
df4=pd.DataFrame(actual_prediction)
df4.to_csv('final_output7_house_price.csv')
