import copy
import json

import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeRegressor

def get_learner(X,y,max_depth=10):
    return DecisionTreeRegressor(max_depth=max_depth).fit(X,y)


def make_trees(X,y,ntrees=100,max_depth=10):
    trees = []
    for i in range(ntrees):
        # Your solution here
        pass
        
    return trees

def make_prediction(trees,X):
    predictions = []
    tree_predictions = []
    for j in range(len(trees)):
        tree = trees[j]
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).mean().values.flat)

def make_trees_boost(Xtrain, Xval, ytrain, yval, max_ntrees=100,max_depth=2):
    #Xtrain, Xval, ytrain, yval = train_test_split(X,y,test_size=val_frac,shuffle=True)
    trees = []
    yval_pred = None
    ytrain_pred = None
    train_RMSEs = [] # the root mean square errors for the validation dataset
    val_RMSEs = [] # the root mean square errors for the validation dataset
    ytrain_orig = copy.deepcopy(ytrain)
    for i in range(max_ntrees):
        # Your solution here
        pass
        
    return trees,train_RMSEs,val_RMSEs

def cut_trees(trees,val_RMSEs):
    # Your solution here that finds the minimum validation score and uses only the trees up to that
    return trees

def make_prediction_boost(trees,X):
    tree_predictions = []
    for tree in trees:
        tree_predictions.append(tree.predict(X).tolist())
    return np.array(pd.DataFrame(tree_predictions).sum().values.flat)

