import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, svm, preprocessing
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
df.drop(['society'], 1, inplace=True)
#print(df.head())
# Y = np.array(df['location'])
# print(Y)
# one_hot_encoder = OneHotEncoder(sparse=True)
# X = np.array(one_hot_encoder.fit_transform(Y))
# print(X)

df = pd.get_dummies(data=df, columns=['area_type', 'availability', 'location'])
# df = df.drop('location', axis=1)
# df = df.drop('area_type', axis=1)
# df = df.join(one_hot_column)

print(df.head())

#print(one_hot_column)

df['total_sqft'] = preprocessing.scale(df['total_sqft']) *100
X = np.array(df.drop(['price'], 1))
y = np.array(df['price'])
y = np.log(y)
#print(df['total_sqft'])
#X = preprocessing.scale(X)
#divide into test and train features
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

# polynomial Linear Regression
# Poly_reg = PolynomialFeatures(degree=2)
# X_poly = Poly_reg.fit_transform(X_train)
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_poly, y_train)
#
#
# X_poly_test = Poly_reg.fit_transform(X_test)
# accuracy = clf.score(X_poly_test, y_test)
# print(accuracy)

# SVM regression
clf1 = LinearRegression()
clf1.fit(X_train, y_train)
accuracy1 = clf1.score(X_test, y_test)
print(accuracy1)
print(y_test, clf1.predict(X_test))

# Find RMSE value
rms1 = mean_squared_error(y_test, clf1.predict(X_test))
print(rms1)



