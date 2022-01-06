import copy

# our standard imports
import numpy as np
import pandas as pd

# of course we need to be able to split into training and test
from sklearn.model_selection import train_test_split

# This is where we can get our models
from sklearn.ensemble import RandomForestClassifier

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import classification_report

def scale(df):
    X = None
    # YOUR SOLUTION HERE
    X = pd.DataFrame(X,columns=df.columns)
    return X

def pca(X,random_state=42):
    columns = ["Change me","Change me"]
    X_pca = None
    # YOUR SOLUTION HERE
    X_pca = pd.DataFrame(X_pca,columns=columns)
    return X_pca

def kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10):
    kmeans_models = {}
    for n_clusters in range_n_clusters:
        # Your solution here
        pass
    return kmeans_models

def assign_labels(X,kmeans_models):
    cluster_labels = {}
    for n_clusters in kmeans_models.keys():
        # Your solution here
        pass
    cluster_labels = pd.DataFrame(cluster_labels)
    return cluster_labels


def silhouette_scores(X,cluster_labels):
    def d(x,y): # For ease of use if you want it
        return np.sqrt(np.sum((x-y)**2))
    a = np.zeros((len(X),))
    b = np.zeros((len(X),))
    s = np.zeros((len(X),))
    # Your solution here
    return s

def bin_x(x,n_clusters=3,random_state=10):
    clusterer = None
    # YOUR SOLUTION HERE
    return clusterer