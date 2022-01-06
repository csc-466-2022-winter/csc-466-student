import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab5.joblib")

# Import the student solutions
import Lab5_helper

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv(
    f"{DIR}/../data/breast_cancer_three_gene.csv",index_col=0
)
X = df.drop('Subtype',axis=1)#.dropna()
t = X['ESR1']
X2 = pd.get_dummies(X.drop('ESR1',axis=1))

m = 1.05

def test_exercise_1():
    ntrials = 50
    RMSEs = []
    for trial in range(ntrials):
        X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.25,random_state=trial)
        trees = Lab5_helper.make_trees(X_train,t_train,ntrees=100)
        y = Lab5_helper.make_prediction(trees,X_test)
        RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
    assert np.mean(answers['exercise_1'])*m > np.mean(RMSEs)

def test_exercise_2():
    ntrials = 50
    RMSEs = []
    for trial in range(ntrials):
        X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.25,random_state=trial)
        X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.25,random_state=trial)
        trees,train_RMSEs,val_RMSEs = Lab5_helper.make_trees_boost(X_train2, X_val, t_train2, t_val, max_ntrees=100)
        trees = Lab5_helper.cut_trees(trees,val_RMSEs)
        y = Lab5_helper.make_prediction_boost(trees,X_test)
        RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
    assert np.mean(answers['exercise_2'])*m > np.mean(RMSEs)