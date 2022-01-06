import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab4.joblib")

# Import the student solutions
import Lab4_helper

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
X = titanic_df2.drop('Cabin',axis=1)#.dropna()
X['CabinLetter'] = X['CabinLetter'].fillna("?")
X['Pclass'] = X['Pclass'].astype(str)
X['SibSp'] = X['SibSp'].astype(str)
X['Parch'] = X['Parch'].astype(str)
X = X.dropna()

X2 = X.copy()
X2['Age'] = pd.cut(X2['Age'],bins=20).astype(str) # bin Age up
X2['Age'].value_counts()

X2['Fare'] = pd.cut(X2['Fare'],bins=20).astype(str) # bin Age up
X2['Fare'].value_counts()

t = titanic_df.loc[X2.index,'Survived']

m = 1000

def test_exercise_1():
    e1 = Lab4_helper.entropy(t)
    e2 = Lab4_helper.entropy(X2['CabinLetter'])
    assert np.round(m*answers['exercise_1'][1]) == np.round(m*e2)

def test_exercise_2():
    g1 = Lab4_helper.gain(t,X2['Sex'])
    g2 = Lab4_helper.gain(t,X2['Pclass'])
    g3 = Lab4_helper.gain(t,X2['Age'])
    assert np.round(m*answers['exercise_2'][2]) == np.round(m*g3)

def test_exercise_3():
    gr1 = Lab4_helper.gain_ratio(t,X2['Sex'])
    gr2 = Lab4_helper.gain_ratio(t,X2['Pclass'])
    gr3 = Lab4_helper.gain_ratio(t,X2['Age'])   
    assert np.round(m*answers['exercise_3'][2]) == np.round(m*gr3)

def test_exercise_4():
    col,gain_ratio = Lab4_helper.select_split(X2,t)
    assert np.round(m*answers['exercise_4'][1]) == np.round(m*gain_ratio)

def test_exercise_5():
    tree = Lab4_helper.make_tree(X2,t)
    assert answers['exercise_5'] == tree

def test_exercise_6():
    tree = Lab4_helper.make_tree(X2,t)
    rules = Lab4_helper.generate_rules(tree)
    def process(rules):
        new_rules = []
        for rule in rules:
            new_rules.append(tuple(rule))
        return set(new_rules)
    assert process(answers['exercise_6']) == process(rules)
    
def test_exercise_7():
    tree2 = Lab4_helper.make_tree2(X,t,min_split_count=20)
    assert answers['exercise_7'] == tree2
    
def test_exercise_8():
    default = 0
    from sklearn.model_selection import train_test_split

    X2_train, X2_test, t_train, t_test = train_test_split(X2, t, test_size=0.3, random_state = 0)
    X_train, X_test = X.loc[X2_train.index], X.loc[X2_test.index]

    tree_id3 = Lab4_helper.make_tree(X2_train,t_train)
    rules_id3 = Lab4_helper.generate_rules(tree_id3)
    tree_c45 = Lab4_helper.make_tree2(X_train,t_train)
    rules_c45 = Lab4_helper.generate_rules(tree_c45)

    y_id3 = X2_test.apply(lambda x: Lab4_helper.make_prediction(rules_id3,x,default),axis=1)
    y_c45 = X_test.apply(lambda x: Lab4_helper.make_prediction(rules_c45,x,default),axis=1)
    
    import Lab3_helper
    
    cm_id3 = Lab3_helper.confusion_matrix(t_test,y_id3,labels=[0,1])
    stats_id3 = Lab3_helper.evaluation(cm_id3,positive_class=1)
    
    cm_c45 = Lab3_helper.confusion_matrix(t_test,y_c45,labels=[0,1])
    stats_c45 = Lab3_helper.evaluation(cm_c45,positive_class=1)
    
    source = pd.DataFrame.from_records([stats_id3,stats_c45])
    source['Method'] = ['ID3','C4.5']
    
    left = answers['exercise_8'].drop('Method',axis=1)
    right = source.drop('Method',axis=1)
    assert np.all(np.round(m*left.values) == np.round(m*right.values))