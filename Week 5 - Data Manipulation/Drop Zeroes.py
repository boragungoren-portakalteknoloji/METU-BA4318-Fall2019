import pandas as pd
import numpy as np
import os
filename = os.getcwd() + "\\Guess Me.csv"
df = pd.read_csv(filename, sep=";") # reads into a dataframe
print("df:")
print(df.count())

# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html

df2 = df.dropna() # Remove any row with at least 1 NA value
# df2 = df.dropna(how='any') # eq. to above
# df2 = df.dropna(how='all') # only rows with all values as NA
print("df2:")
print(df2.count())

df3 = df.dropna(thresh=2) # Keep rows with at least 2 NA values
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
print("df3:")
print(df3.count())

df.dropna(how='any',inplace=True) # Make changes on the data frame itself
print("Inplace df:")
print(df.count())


