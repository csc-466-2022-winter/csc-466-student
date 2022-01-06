import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_FinalLab.joblib")

# Import the student solutions
import FinalLab_helper

import pandas as pd
import numpy as np

titanic_df = pd.read_csv(
    f"{DIR}/../data/titanic.csv"
)
features = ['Pclass','Survived','Sex','Age']
titanic_df = titanic_df.loc[:,features]
titanic_df.loc[:,'Pclass']=titanic_df['Pclass'].fillna(titanic_df['Pclass'].mode()).astype(int)
titanic_df.loc[:,'Age']=titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df.loc[:,'Age']=(titanic_df['Age']/10).astype(str).str[0].astype(int)*10
titranic_df = titanic_df.dropna()

df = pd.read_csv(f"{DIR}/../data/breast_cancer_three_gene.csv",index_col=0)

m=1000
def test_exercise_1():
    X = titanic_df.drop('Survived',axis=1)
    y = titanic_df['Survived']
    clf = FinalLab_helper.NBClassifier()
    clf.fit(X,y)
    predictions = clf.predict(X)
    assert np.all(np.round(m*answers['exercise_1']) == np.round(m*predictions))

def test_exercise_2():
    X = df[['ESR1','AURKA']]
    pca_transformer = FinalLab_helper.PCA()
    pca_transformer.fit(X)
    Xt = pca_transformer.transform(X)
    assert np.all(np.round(m*answers['exercise_2']) == np.round(m*Xt))

