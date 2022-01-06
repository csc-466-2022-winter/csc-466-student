import copy

# our standard imports
import numpy as np
import pandas as pd

# of course we need to be able to split into training and test
from sklearn.model_selection import train_test_split

def predict_user_user(data_raw,x_raw,N=10,frac=0.02):
    # data_raw is our uncentered data matrix. We want to make sure we drop the name of the user we
    # are predicting:
    db = data_raw.drop(x_raw.name)
    # We of course want to center and fill in missing values
    db = (db.T-db.T.mean()).fillna(0).T
    # Now this is a little tricky to think about, but we want to create a train test split of the movies
    # that user x_raw.name has rated. We need some of them but want some of them removed for testing.
    # This is where the frac parameter is used. I want you to think about how to select movies for training
    # Your solution here
    #ix_raw, ix_raw_test = train_test_split(???,test_size=frac,random_state=42) # Got to ignore some movies
    
    # Here is where we use what you figured out above
    x_raw_test = x_raw.loc[ix_raw_test] 
    x_raw = x_raw.copy()
    x_raw.loc[ix_raw_test] = np.NaN # ignore the movies in test
    x = (x_raw - x_raw.mean()).fillna(0)

    preds = []
    for movie in ix_raw_test:
        # Your solution here
        #sims = db.loc[??? Only look at users who have rated this movie ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        try:
            sorted_sims = sims.sort_values()[::-1]
        except:
            preds.append(0) # means there is no one that also rated this movie amongst all other users
            continue
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        # Your solution here
        #preds.append(??? using ids how do you predict ???)
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test-x_raw.mean()
    mae = (actual-pred).abs().mean()
    return mae

def predict_item_item(data_raw,x_raw,N=10,frac=0.02,debug={}):
    ix_raw, ix_raw_test = train_test_split(x_raw.dropna().index,test_size=frac,random_state=42) # Got to ignore some movies
    x_raw_test = x_raw.loc[ix_raw_test]
    
    db = data_raw.drop(x_raw.name)
    db = (db.T-db.T.mean()).fillna(0).T
    # ??? db = FIX DB SO WE CAN KEEP CODE SIMILAR BUT DO ITEM-ITEM ???
    preds = []
    for movie in ix_raw_test:
        x = db.loc[movie]
        # sims = db.drop(movie).loc[??? ONLY SELECT MOVIES IN TRAINING SET WHICH USER HAS RATED ???].apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
        sims = sims.dropna()
        sorted_sims = sims.sort_values()[::-1]
        top_sims = sorted_sims.iloc[:N]
        ids = top_sims.index
        #preds.append(??? HOW TO PREDICTION ???)
    pred = pd.Series(preds,index=x_raw_test.index)
    actual = x_raw_test
    mae = (actual-pred).abs().mean()
    return mae
