import pandas as pd
import numpy as np
import os
import psutil
# psutil is used to track memory usage
# https://psutil.readthedocs.io/en/latest/
import time
# time is used to measure wall-clock time

# Functions for Question 3
def convert_number_to_KMG(x):
    if x < 1000:
        x = round(x,-1)
        return str(x)
    elif x < 1000000:
        x = round(x,-2) / 1000
        return str(x)+"K"
    elif x < 1000000000:
        x = round(x,-5) / 1000000
        return str(x)+"M"
    else:
        x = round(x,-8) / 1000000000
        return str(x)+"G"

def read_file(cityname, filename, columns, names, test):
    city_file = os.getcwd() + "\\" + filename
    city_cols = ["CET", "Mean TemperatureC"]
    df_city = None
    if test == True:        
        df_city = pd.read_csv(city_file, usecols=columns, nrows=10000)
    else:
        df_city = pd.read_csv(city_file, usecols=columns)
    df_city.columns = names
    print(cityname, " has ", len(df_city), " records.")
    #print(df_city)
    return df_city

# Functions for Question 4
def resample_means(data_frame, cols, mtime=False):
    # cols[0] is the column for Date values
    # cols[1] is the column for samples (to be resampled)
    start = None
    end = None
    if mtime == True:
        start = time.time()
    rows_list = []
    for day in data_frame[cols[0]].unique():
        # calculate mean for all values recorded on that day
        dict = {}
        mean = round( data_frame.loc[ data_frame[cols[0]] == day ][cols[1]].mean(), 1)
        dict.update( { cols[0]:day, cols[1]:mean} )
        rows_list.append(dict)
    df_ret = pd.DataFrame(rows_list)
    if mtime == True:
        end = time.time()
        duration = round( end - start, 1)
        print("Summary took", duration, " seconds.")
    return df_ret

def resample_means_2(data_frame, cols, mtime=False):
    # cols[0] is the column for Date values
    # cols[1] is the column for samples (to be resampled)
    start = None
    end = None
    if mtime == True:
        start = time.time()
    rows_list = []
    for day in data_frame[cols[0]].unique():
        # calculate mean for all values recorded on that day
        dict = {}
        mean = round( data_frame.loc[ data_frame[cols[0]] == day ][cols[1]].mean(), 1)
        dict.update( { cols[0]:day, cols[1]:mean} )
        rows_list.append(dict)
    df_ret = pd.DataFrame(rows_list)
    if mtime == True:
        end = time.time()
        duration = round( end - start, 1)
        print("Summary took", duration, " seconds.")
    return df_ret


def process_na_values(df, approach="drop"):
    if approach == "drop":
        df.dropna(how="any",inplace=True)
    # you can add more approaches, but in the HW it is
    # strictly specified as "drop any"
    elif approach == "fill":
        df.fillna(method="ffill",inplace=True)
    elif approach == "interpolate":
        df.interpolate(method="cubic",inplace=True)
    else:
        df = df # dummy assignment to stress do nothing
    return df  

# This one is more hands-on Python
def merge_dataframes(df_left, date_left, df_right, date_right, na_approach="drop", mtime=False):
    start = None
    end = None
    if mtime == True:
        start = time.time()
    df_merged = pd.merge(df_left, df_right, left_on=date_left, right_on=date_right,how="outer")
    df_merged = process_na_values(df_merged, approach=na_approach)
    df_merged.rename({date_left:"date"}, inplace=True)
    del df_merged[date_right]
    df_merged = df_merged.reset_index(drop=True)
    if mtime == True:
        end = time.time()
        duration = round( end - start, 1)
        print("Merge took", duration, " seconds.")
    return df_merged

# This one makes a little more use of pandas dataframe features
# for a manual inner join
def merge_dataframes_2(df_left, date_left, df_right, date_right, na_approach="drop", mtime=False):
    start = None
    end = None
    if mtime == True:
        start = time.time()
    df_merged = None
    # Filter rows in the left table where dates that
    # do not appear in the right one are gone
    df_left_filtered = df_left.where(df_left[date_left].isin(list(set(df_right[date_right])))).dropna().sort_values(by=[date_left]).reset_index(drop=True)
    # Filter rows in the right table where dates that
    # do not appeat in the left one are gone
    df_right_filtered = df_right.where(df_right[date_right].isin(list(set(df_left[date_left])))).dropna().sort_values(by=[date_right]).reset_index(drop=True)
    # At this point both tables should have only the
    # intersecting dates and they should also be sorted by date
    # We simply concat
    df_merged = pd.concat([df_left_filtered, df_right_filtered], axis=1, join="inner")
    # Now we just rename one date column to "date" and delete the other one.
    df_merged.rename({date_left:"date"}, inplace=True)
    del df_merged[date_right]
    # reset the indexes 
    df_merged = df_merged.reset_index(drop=True)
    if mtime == True:
        end = time.time()
        duration = round( end - start, 1)
        print("Merge took", duration, " seconds.")
    return df_merged

# Question 3 - Extracting the dataset
# You should be using a few hundred megabytes of memory for this program, depending on paging. 
# Uncomment the following lines to see memory usage
# memused = convert_number_to_KMG ( psutil.virtual_memory().used)
# memfree = convert_number_to_KMG ( psutil.virtual_memory().free) 
# print("Before loading files, system memory used in this computer:", memused)
# print("Before loading files, system memory free in this computer:", memfree)

df_madrid = read_file("Madrid", "weather_madrid_LEMD_1997_2015.csv", ["CET", "Mean TemperatureC"],["date-madrid", "temp-madrid"], test=False)
df_brazil = read_file("Brazil", "sudeste.csv", ["date", "temp"],["date-brazil", "temp-brazil"], test=False)

# Uncomment the following lines to see memory usage
# memused = convert_number_to_KMG ( psutil.virtual_memory().used)
# memfree = convert_number_to_KMG ( psutil.virtual_memory().free) 
# print("After loading files, system memory used in this computer:", memused)
# print("After loading files, system memory free in this computer:", memfree)

# Question 4 - Transforming the dataset
# Create summary for Brazil
columns = ["date-brazil", "temp-brazil"]
df_brazil = resample_means(df_brazil, columns,mtime=True)
print("Brazil summary has ", len(df_brazil), " records.")
# Merge two datasets
df_merged = merge_dataframes(df_madrid, "date-madrid", df_brazil, "date-brazil", na_approach="drop", mtime=True)
print("Merged data set has ", len(df_merged), " records.")
# print(df_merged)

# Individual data sets have no use now, uncomment the following lines to save memory
# df_madrid = None
# df_brazil = None

# Question 5 - See Correlation
print(df_merged.corr(method ='pearson'))
# Correlation ratio is -0.030 ~ 0

# Question 6 - Explanation
# There is a very small negative correlation.
# 1. The data set includes periodically increasing and decreasing values
# because of seasons (spring, summer, etc.) Because of this alternating
# nature, correlation is expected to be near-zero. 
# 2. A possible reason for a non-zero value is a trend in both series.
# Note that a "trend" should be irrespective of seasons and have a
# positive/negative direction. 
# 3. Furthermore, these locations are in different hemispheres. This means
# when one has winter, the other has summer. Hence the correlation ratio
# is negative.

