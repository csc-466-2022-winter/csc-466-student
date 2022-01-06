import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab7.joblib")

# Import the student solutions
import Lab7_helper

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

ratings = pd.read_csv(f'{DIR}/../data/movielens-small/ratings.csv') # you might need to change this path
ratings = ratings.dropna()
movies = pd.read_csv(f'{DIR}/../data/movielens-small/movies.csv')
movies = movies.dropna()
data = movies.set_index('movieId').join(ratings.set_index('movieId'),how='inner').reset_index()
data = data.drop('timestamp',axis=1) # We won't need timestamp here
ratings = data.set_index(['userId','movieId'])['rating']

m=1000
def test_exercise_1():
    mae = Lab7_helper.predict_user_user(ratings.unstack(),ratings.unstack().loc[1])
    assert np.round(m*answers['exercise_1']) == np.round(m*mae)

def test_exercise_2():
    mae = Lab7_helper.predict_item_item(ratings.unstack(),ratings.unstack().loc[1])
    assert np.round(m*answers['exercise_2']) == np.round(m*mae)
