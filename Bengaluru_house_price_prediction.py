import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, svm, preprocessing
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
#print(df.head())

## ENCODE NON-NUMERIC DATA TO NUMERIC VALUES.

#df.drop(['society'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)   # 0 and not -99999 coz age, sex etc cant be that value
#print(df.head())

def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        # create empty dictionary, {'female':0, 'male':1} like that where, the 0,1 are returned thus converting them into integers
        def convert_to_int(key):
            return text_digit_vals[key]


        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents= df[column].values.tolist()
            unique_elements = set (column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))    # altering the df values accordingly to the required numeric data set

    return df

df = handle_non_numeric_data(df)
#print(df.head)

## NON - NUMERIC DATA CLEANED


#df.drop(['location', 'society', 'availability', 'area_type'], 1, inplace=True)

X = np.array(df.drop(['price'], 1))
y = np.array(df['price'])
# y= np.log(y)
X = preprocessing.scale(X)
# df['total_sqft'] = preprocessing.scale(df['total_sqft']) *100
# divide into test and train features
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)
X[:, 5] = X[:, 5] * 100
# print(X[:, 5])

# Simple Linear Regression
clf1 = LinearRegression()
clf1.fit(X_train, y_train)
accuracy1 = clf1.score(X_test, y_test)
print('linear regression score =', accuracy1)

# Support Vector Machines - Linear SVR
clf2 = svm.LinearSVR()
clf2.fit(X_train, y_train)
accuracy2 = clf2.score(X_test,y_test)
print('SVM.LinearSVR score =', accuracy2)

# Support Vector Machines - SVR
clf3 = svm.SVR()
clf3.fit(X_train, y_train)
accuracy3 = clf3.score(X_test,y_test)
print('SVM.SVR score =', accuracy3)

# polynomial Linear Regression    # degree 5 is giving good results for full X preprocessing
Poly_reg = PolynomialFeatures(degree=5, interaction_only=False, include_bias=False)
X_poly = Poly_reg.fit_transform(X_train, y=5)
clf = LinearRegression(fit_intercept= True, normalize= False, n_jobs=-1)
clf.fit(X_poly, y_train)
X_poly_test = Poly_reg.fit_transform(X_test)
# accuracy = clf.score(X_poly_test, y_test)
accuracy = r2_score(y_test, clf.predict(X_poly_test), multioutput='variance_weighted')
print('poly regression score=', accuracy)

#print('Poly_reg', y_test, clf.predict(X_poly_test))
#print('Linear_regression', y_test, clf1.predict(X_train))

#rms1 = mean_squared_error(clf.predict(X_poly_test), y_test)
#print('rms_poly', rms1)

#df1 = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Test-Data.csv')
#print(df1.head)

