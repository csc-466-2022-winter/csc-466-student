import sys
import os
sys.path.append(".")

import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pathlib
DIR=pathlib.Path(__file__).parent.absolute()

import joblib 
answers = joblib.load(str(DIR)+"/answers_Lab1.joblib")

# Import the student solutions
import Lab1_helper as helper

import numpy as np
np.random.seed(1)
c = np.random.rand(6,4)

import pandas as pd
titanic_df = pd.read_csv(
    f"{DIR}/../data/titanic.csv"
)

credit = pd.read_csv(f"{DIR}/../data/credit.csv",index_col=0)

def test_exercise_1():
    assert np.all(answers['exercise_1'] == helper.exercise_1())

def test_exercise_2():
    assert np.all(answers['exercise_2'] == helper.exercise_2())

def test_exercise_3():
    assert answers['exercise_3'] == helper.exercise_3(c)

def test_exercise_4():
    one,two = helper.exercise_4(c)
    assert np.all(answers['exercise_4'][0] == one) and np.all(answers['exercise_4'][1] == two)
    
def test_exercise_5():
    b = helper.exercise_2()
    one,two = helper.exercise_5(b)
    assert np.all(answers['exercise_5'][0] == one) and np.all(answers['exercise_5'][1] == two)
    
def test_exercise_6():
    assert np.all(answers['exercise_6'] == helper.exercise_6())

def test_exercise_7():
    assert np.all(answers['exercise_7'] == helper.exercise_7())
    
def test_exercise_8():
    sel = helper.exercise_8(titanic_df)
    assert np.all(answers['exercise_8'] == sel)
    
def test_exercise_9():
    sel = helper.exercise_9(titanic_df)
    assert answers['exercise_9'].equals(sel)

def test_exercise_10():
    titanic_df_copy = titanic_df.set_index('Name')
    helper.exercise_10(titanic_df_copy)
    assert answers['exercise_10'].equals(titanic_df_copy)
    
def test_exercise_11():
    X = credit.drop(['Gender','LoanAmountApproved'],axis=1)
    y = credit['LoanAmountApproved']
    model = helper.exercise_11(X,y)

    assert np.all(answers['exercise_11'] == model.coef_)