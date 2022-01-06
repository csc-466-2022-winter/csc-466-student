import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab3.joblib")

# Import the student solutions
import Lab3_helper

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def truncate(d, mult=10000):
    for k in d.keys():
        d[k] = np.round(d[k]*mult)/mult
    return d


titanic_df = pd.read_csv(
    f"{DIR}/../data/titanic.csv"
)

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
titanic_df2 = titanic_df.loc[:,features]
titanic_df2['CabinLetter'] = titanic_df2['Cabin'].str.slice(0,1)
X = pd.get_dummies(titanic_df2.drop('Cabin',axis=1)).dropna()
# Standard scaling
means = X.mean()
sds = X.std()
X2 = X.apply(lambda x: (x-means)/sds,axis=1)
t = titanic_df.loc[X2.index,'Survived']

def test_exercise_1():
    seeds = [0,1,2,3,4,5]
    results = pd.DataFrame(columns=["seed","frac_max_class","accuracy_test","accuracy_train2","accuracy_val"]).set_index("seed")
    for seed in seeds:
        np.random.seed(seed)
        X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.3)
        X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.3)

        ## Your solution in evaluate_baseline(...)
        frac_max_class,accuracy_test,accuracy_train2,accuracy_val = Lab3_helper.evaluate_baseline(t_test,t_train2,t_val)

        results = results.append(pd.Series([frac_max_class,accuracy_test,accuracy_train2,accuracy_val],index=results.columns,name=seed))
    m = 1000
    assert np.all(np.round(m*answers['exercise_1'].values) == np.round(m*results.values))

def test_exercise_2():
    w,X_test,t_test,results = Lab3_helper.train(X2,t,seed=0)
    y_test = Lab3_helper.predict(w,X_test)
    assert np.all(answers['exercise_2'].values == y_test.values)

def test_exercise_3():
    w,X_test,t_test,results = Lab3_helper.train(X2,t,seed=0)
    y_test = Lab3_helper.predict(w,X_test)
    cm = Lab3_helper.confusion_matrix(t_test,y_test,labels=[0,1])    
    assert np.all(answers['exercise_3'].values == cm.values)

def test_exercise_4():
    w,X_test,t_test,results = Lab3_helper.train(X2,t,seed=0)
    y_test = Lab3_helper.predict(w,X_test)
    cm = Lab3_helper.confusion_matrix(t_test,y_test,labels=[0,1]) 
    student = (Lab3_helper.evaluation(cm,positive_class=1),Lab3_helper.evaluation(cm,positive_class=0))
    assert truncate(answers['exercise_4'][0]) == truncate(student[0]) and truncate(answers['exercise_4'][1]) == truncate(student[1])

def test_exercise_5():
    seeds = [0,1,2,3,4,5]
    importances = Lab3_helper.importance(X2,t,seeds)
    m = 1000
    assert np.all(np.round(m*answers['exercise_5'].values) == np.round(m*importances.values))