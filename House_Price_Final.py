import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlwt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation, svm, preprocessing
from sklearn.metrics import mean_squared_error

df = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Train-Data.csv')
df1 = df1 = pd.read_csv('Predicting-House-Prices-In-Bengaluru-Test-Data.csv')
df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)   # 0 and not -99999 coz age, sex etc cant be that value

def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        # create empty dictionary, {'female':0, 'male':1} like that where, the 0,1 are returned thus converting them into integers
        def convert_to_int(key):
            return text_digit_vals[key]


        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents= df[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))    # altering the df values accordingly to the required numeric data set

    return df

df = handle_non_numeric_data(df)
df1 = handle_non_numeric_data(df1)

# Train data features
X = np.array(df.drop(['price'], 1))
y = np.array(df['price'])
# y= np.log(y)
X = preprocessing.scale(X)
X[:, 5] = X[:, 5] * 100

# Test data features
X1 = np.array(df1.drop(['price'], 1))
y1 = np.array(df1['price'])
X1 = preprocessing.scale(X1)
X1[:, 5] = X1[:, 5] * 100

# Train classifier
# polynomial Linear Regression    # degree 5 is giving good results for full X preprocessing
Poly_reg = PolynomialFeatures(degree=5, interaction_only=False, include_bias=False)
X_poly = Poly_reg.fit_transform(X, y=5)
clf = LinearRegression(fit_intercept= False, normalize= False, n_jobs=-1)
clf.fit(X_poly, y)


X_poly_test = Poly_reg.fit_transform(X1)
Missing_Prices = clf.predict(X_poly_test)
Missing_Prices = np.array(Missing_Prices)
df2 = pd.DataFrame(Missing_Prices)
print(abs(Missing_Prices))
df.drop(['price'], 1, inplace=True)
df.join(df2)
print(df.head())
df2.to_excel('test_house_price_bangalore.xls', index = False)

