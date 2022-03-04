import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab6.joblib")

# Import the student solutions
import Lab6_helper

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import pandas as pd
df = pd.read_csv(f"{DIR}/../data/housing/boston_fixed.csv")

def test_exercise_1(run_assert=True):
    X = Lab6_helper.scale(df)
    if run_assert == False:
        return X
    assert np.all(answers['exercise_1'].values == X.values)

def test_exercise_2(run_assert=True):
    X = Lab6_helper.scale(df)
    X_pca = Lab6_helper.pca(X)
    if run_assert == False:
        return X_pca
    m = 100
    assert np.all(np.round(m*answers['exercise_2'].values) == np.round(m*X_pca.values))
    
def test_exercise_3(run_assert=True):
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    if run_assert == False:
        return kmeans_models
    assert set(answers['exercise_3'].keys()) == set(kmeans_models.keys())

def test_exercise_4(run_assert=True):
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    cluster_labels = Lab6_helper.assign_labels(X,kmeans_models)
    if run_assert == False:
        return cluster_labels
    assert np.all(answers['exercise_4'].values == cluster_labels.values)
    
def test_exercise_5(run_assert=True):
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    cluster_labels = Lab6_helper.assign_labels(X,kmeans_models)
    n_clusters = 2
    m = 1000
    scores = Lab6_helper.silhouette_scores(X,cluster_labels[n_clusters])
    if run_assert == False:
        return scores
    assert np.all(np.round(m*answers['exercise_5']) == np.round(m*scores))
    
def test_exercise_6(run_assert=True):
    X = Lab6_helper.scale(df)
    kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
    cluster_labels = Lab6_helper.assign_labels(X,kmeans_models)
    n_clusters = 2
    scores = Lab6_helper.silhouette_scores(X,cluster_labels[n_clusters])
    clusterer = Lab6_helper.bin_x(df[["MEDV"]])
    labels = clusterer.predict(df[["MEDV"]])
    if run_assert == False:
        return labels
    assert np.all(answers['exercise_6'] == labels)