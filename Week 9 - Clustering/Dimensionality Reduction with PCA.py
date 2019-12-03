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
# PCA, K-Means and DBSCAN related
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Dimensionality Reduction With Principal Component Analysis (PCA)
# (PCA) is a linear dimensionality reduction technique that can be
# utilized for extracting information from a high-dimensional space
# by projecting it into a lower-dimensional sub-space. It tries to
# preserve the essential parts that have more variation of the data
# and remove the non-essential parts with fewer variation.

# From a pure regression point of view, PCA identifies and removes
# dependent variables from the dataset.

# Because PCA reduces dimensions it makes it easier to visualize data
# It also speeds up many techniques by removing redundant parts of your dataset

def read_file(filename, columns, s=","):
    file = os.getcwd() + "\\" + filename
    df = pd.read_csv(file, sep=s, usecols=columns)
    return df

# The following file has X1 to X3 as original values
# and X4 to X6 as derived values. 
filename="Sample with Derived Columns.csv"
cols = ["X1", "X2", "X3", "X4", "X5", "X6"]
df = read_file(filename, columns=cols, s=";")

# Step 0 - Scale the dataset using minmax scaler
df_ndarray = MinMaxScaler(feature_range=[0, 1]).fit_transform(df)

# Step 1a - Apply CSEV method to understand desired number of dimensions
# pca = PCA(n_components=2)
pca = PCA()
principalComponents = pca.fit_transform(df_ndarray)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Dataset Explained Variance')
plt.show()
# Plot show N=3 is better but N=2 is almost the same

# Step 1b - Apply Elbow method to understand number of clusters
def show_wccs(df, title="Elbow Method"):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(df)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title(title)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

show_wccs(df,"Elbow Method before PCA")

# Step 1c - Apply Silhouette method to understand number of clusters
def show_sil(df, title="Silhouette Method"):
    sil = []
    # dissimilarity would not be defined for a single cluster.
    # Thus, minimum number of clusters should be 2
    for k in range(2, 11):
      kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
      kmeans.fit(df)
      labels = kmeans.labels_
      sil.append(silhouette_score(df, labels, metric = 'euclidean'))
    plt.plot(range(2, 11), sil)
    plt.title(title)
    plt.xlabel("Number of clusters")
    plt.ylabel("Sil Score")
    plt.show()

show_sil(df, title="Silhouette Method before PCA")
# Both 1b and 1c say there are 4 clusters
# Step 2 - Now apply PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(df_ndarray)
dfPCA = pd.DataFrame(data = principalComponents, columns = ["PC1", "PC2"])
# Step 1b - Plot the components
plt.scatter(dfPCA.PC1, dfPCA.PC2)
plt.title("2 component PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Step 3a - Now apply K-Means to this dataset
show_wccs(df=dfPCA,title="Elbow Method after PCA")
show_sil(df=dfPCA, title="Silhouette Method after PCA")
# Both methods shows there are 6 clusters
N=6

def cluster_kmeans(df, N):   
    kmeans = KMeans(n_clusters=N, init='k-means++', max_iter=300, n_init=10, random_state=0)
    labels = kmeans.fit_predict(df)
    centers_x = kmeans.cluster_centers_[:, 0]
    centers_y = kmeans.cluster_centers_[:, 1]
    return labels, centers_x, centers_y

labels, centers_x, centers_y=cluster_kmeans(df=dfPCA,N=N)
plt.scatter(dfPCA.PC1, dfPCA.PC2, c=labels, cmap='viridis', edgecolor='k')
plt.title("KMEANS - Estimated number of clusters: %d" % N)
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.scatter(centers_x, centers_y, s=300, c='red')
plt.show()

# Step 3b - Now apply DBSCAN to this dataset
def cluster_dbscan(df):
    X = StandardScaler().fit_transform(df)
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    noise = list(labels).count(-1)
    noiseExists = 0
    if noise > 0:
        noiseExists = 1
    clusters = len(set(labels)) - noiseExists
    print('Estimated number of clusters: %d' % clusters)
    print('Estimated number of noise points: %d' % noise)
    return labels, clusters, noise

labels, clusters, noise = cluster_dbscan(df=dfPCA)
plt.scatter(dfPCA.PC1, dfPCA.PC2, c=labels, cmap='viridis', edgecolor='k')
# For alternate colormaps
# https://matplotlib.org/examples/color/colormaps_reference.html
plt.title("DBSCAN - Estimated number of clusters: %d" % clusters)
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()