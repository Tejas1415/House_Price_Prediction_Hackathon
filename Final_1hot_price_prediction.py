# -*- coding: utf-8 -*-
"""
Created on Sun May 27 22:35:31 2018

@author: Tejas_2

My second submission to the hackathon, 85% accuracy found. 13th place.
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

# Input the test and Train datasets
df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
df.drop(['society'], 1, inplace=True)
df.drop(['availability'],1, inplace=True)
#df = df.fillna(0)

df1 = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Test-Data.csv')
df1.drop(['society', 'availability', 'price'], 1, inplace=True)
#df1 = df1.fillna(0)


###################################################################################
# The below part of the code must be run seperately, to balance the features since-
# -we are using2 different datasets for test and train. A location in df might not be in df2- 
# -and vice versa.
# Finding the new features involved
locations = np.array(df.location.unique())
locations_test = np.array(df1.location.unique())
locations =df.location.unique()

# Manual work, calculate missing items and manually add them to the csv file.
# Missing elements in train dataset, add the resultant elements to train_locations column.
x = (set(locations_test)-set(locations))
x = list(x)
print(x)
df2=pd.DataFrame(x)
df2.to_csv('out.csv')
print(len(x))

# Missing elements in test dataset, add them to test database locations column.
y= (set(locations)-set(locations_test))  
y=list(y)
print(len(y))
##################################################################################




# One-Hot encode the location column
df = pd.get_dummies(data=df, columns=['location'])
df1 = pd.get_dummies(data=df1, columns=['location'])

# Once encoded drop the unwantedly, manually added columns
# The manual addition was just to get correct features during one hot encoding.
df = df.dropna(axis=0)
df1 = df1.dropna(axis=0)
# Convert them to computable arrays 
X = np.array(df.drop(['price'], 1))
y = np.array(df['price'])
#y = np.log(y)

X_test = np.array(df1)

# Train the classifier
clf1 = GradientBoostingRegressor(alpha=0.7, n_estimators=200, max_depth = 15, min_samples_split = 2,
                                 learning_rate = 0.5, max_features=200, subsample = 1, 
                                 random_state=3, loss='ls')

# Fit the trained model to the dataset
clf1.fit(X,y)

# Now predict the price values in the test dataset.
prediction = clf1.predict(X_test)
df4=pd.DataFrame(prediction)
df4.to_csv('final_output2_house_price.csv')
# Calculate the mean square error
score = mean_squared_error(y_test, prediction)
print(score)

