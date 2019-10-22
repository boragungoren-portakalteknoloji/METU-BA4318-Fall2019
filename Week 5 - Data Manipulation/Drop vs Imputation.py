import pandas as pd
import numpy as np
import os

def prepare_file(method):
    filename = os.getcwd() + "\\Guess Me.csv"
    df = pd.read_csv(filename, sep=";") # reads into a dataframe
    if method == 'drop-any':
        df.dropna(how='any',inplace=True) # Make changes on the data frame itself
    elif method == 'drop-all':
        df.dropna(how='all',inplace=True) # Make changes on the data frame itself
    elif method == 'forward-fill':
        df.fillna(method='ffill', inplace=True)
    elif method == 'backward-fill':
        df.fillna(method='bfill', inplace=True)
    else:
        # assume any
        df.dropna(how='any',inplace=True)
    return df

def test_methods():
    method_strings = ['drop-any', 'drop-all', 'forward-fill', 'backward-fill' ]
    for method_name in method_strings:
        print ("Using method: ", method_name)
        df = prepare_file(method_name)
        # print averages for all columns
        print("Column means")
        print( df.mean() )
    # done with for loop
    return

# test_methods()
df = prepare_file('drop-any')
# play with different methods to observe effect

# import two similar libraries to do linear regression
from sklearn import linear_model as lm
import statsmodels.api as sm

Y = df[['Y']] # dependent variable
X = df[['X1','X2','X3']] # selected independent variables
# Linear Regression
print("Simple Linear Regression")
regr = lm.LinearRegression()
regr.fit(X, Y)
print('Intercept: \n', regr.intercept_)
print('Coefficients: \n', regr.coef_)

# Better model with constant
print("Regression Model Details")
X = sm.add_constant(X) # adding a constant
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
print_model = model.summary()
print(print_model)