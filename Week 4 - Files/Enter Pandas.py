import pandas as pd
import numpy as np
import os
filename = os.getcwd() + "\\dataset.txt"
df = pd.read_csv(filename) # reads into a dataframe
print(df)
#print(df.axes)
#print(df.index)
#print(df.columns)

# access individual column
#print ("The name column.")
# print(df["Name"])
#print("Has size:", df.Name.size, " , has objects of type", df.Name.dtype )
print("Unique names:", df.Name.unique())
print("Frequencies:")
print(df.Name.value_counts())


#print("The coffee consumption")
#print("Average:", df.Cup.mean())
#print("The first two lines have data:")
#print(df.Cup.head(2) )
#print("The last two lines have data:")
#print(df.Cup.tail(2) )

#coffeearray = np.asarray(df.Cup)
#print(coffeearray)
# [7 3 4 4 3 0]

#print( df.groupby("Name")["Cup"].sum() )
#             ^ Filter
#for name in df.Name.unique():
#    print(name)

