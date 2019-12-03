#generic
import pandas as pd
import numpy as np
import os
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")
# plotting related
import matplotlib.pyplot as plt
import seaborn as sns# for plot styling
sns.set()  
# kmeans related
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def read_file(filename, columns, s=","):
    file = os.getcwd() + "\\" + filename
    df = pd.read_csv(file, sep=s, usecols=columns)
    return df

filename="Sample with Three Clusters.csv"
#filename="Sample With Outliers.csv"
filename="Sample with Scale Issues.csv"
df = read_file(filename, ["X", "Y"], s=";")
# Step 1 - Check for scaling problems
# On this dataset we have no scaling problems.
print("Describing X and Y values")
print( df.describe())
# print( df.Y.describe())
input("Press Enter to continue...")

# Step 2 - Plot the values for some more insight since we have only 2 dimensions.
# This would be impossible for more dimensions
def show_scatter(df):
    plt.scatter(df.X,df.Y)
    plt.title("Sample data has obvious clusters")
    plt.xlabel("X Values")
    plt.ylabel("Y Values")
    plt.show()
show_scatter(df)

# Now let's do this with K-means
# Step 3 - Preprocess data with standard scaler
# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
# If we had scaling problems we would scale the dataset
# scaler = StandardScaler()
# df_ndarray = scaler.fit_transform(df)
df_ndarray = df # comment this line if you choose to scale

# Step 4a - Estimate the number of clusters "visually"
# Variant: Elbow Method
# Within Cluster Sum of Squares (WCSS) method shows us an "elbow"
# where the change in WCSS begins to level off, so that
# increasing number of clusters does not matter
# The Squared Error for each point is the square of the distance of the point
# from its representation i.e. its predicted cluster center.
# The WSS score is the sum of these Squared Errors for all the points.
# Any distance metric like the Euclidean Distance or the Manhattan Distance can be used.
def show_wccs(df):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

show_wccs(df_ndarray)
# For the sample dataset,
# WCCS should show that N=3 clusters will be better. N=2 might also be tried
N=3

# Step 4b - Estimate the number of clusters "visually"
# Variant: Silhouette Method
# The silhouette value measures how similar a point is to its own cluster
# (cohesion) compared to other clusters (separation)
# The range of the Silhouette value is between +1 and -1.
# A high value is desirable and indicates that the point is placed in the correct cluster.
# If many points have a negative Silhouette value,
# it may indicate that we have created too many or too few clusters.
def show_sil(df):
    sil = []
    # dissimilarity would not be defined for a single cluster.
    # Thus, minimum number of clusters should be 2
    for k in range(2, 11):
      kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
      kmeans.fit(df)
      labels = kmeans.labels_
      sil.append(silhouette_score(df, labels, metric = 'euclidean'))
    plt.plot(range(2, 11), sil)
    plt.title('Silhouette Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Sil Score')
    plt.show()

show_sil(df_ndarray)

# Step 6 - Create cluster assignments
N=3
#N=4 # For the best score in sil method with the outlier dataset
def cluster_kmeans(df, N):   
    kmeans = KMeans(n_clusters=N, init='k-means++', max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit_predict(df)
    centers_x = kmeans.cluster_centers_[:, 0]
    centers_y = kmeans.cluster_centers_[:, 1]
    return labels, centers_x, centers_y

labels, centers_x, centers_y=cluster_kmeans(df,N)
# here the output is the assignments on the labeled clusters
# Uncomment the line below to see that the assignments are integers like 0,1,2
# print(labels)
# Let's display the cluster centers
plt.scatter(df.X, df.Y, c=labels, cmap='viridis', edgecolor='k')
# For alternate colormaps
# https://matplotlib.org/examples/color/colormaps_reference.html
plt.title("Cluster centers on original data")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.scatter(centers_x, centers_y, s=300, c='red')
plt.show()
input("Done.")