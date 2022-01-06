# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] slideshow={"slide_type": "slide"} hideCode=false hidePrompt=false
# # Chapter 5 - Decision by Committee
#
# Paul E. Anderson

# + slideshow={"slide_type": "skip"}
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system

# + [markdown] slideshow={"slide_type": "subslide"}
# <img src="https://s36369.pcdn.co/wp-content/uploads/2018/11/Two-heads-are-better-than-one-HP.jpg">

# + [markdown] slideshow={"slide_type": "subslide"}
# **Definition:** Ensemble learning is a type of learning that combines a set/ensemble of learners to reach a prediction.

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Two main types of ensemble learning
#
# <img src="https://miro.medium.com/max/2000/1*zTgGBTQIMlASWm5QuS2UpA.jpeg">

# + [markdown] slideshow={"slide_type": "subslide"}
# One thing all these methods have to decide is how to get different models from the same dataset. Bagging gets the different models by dividing up the training data. Boosting gets different models by training a series of models on the residual of the previous model.

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Bagging
# Bagging stands for bootstrap aggregating. 
#
# So what is a bootstrap? A bootstrap sample is a sample taken from the original dataset with replacement. A new dataset created with boostrap sampling is the same size as the original. 
#
# This is completely different from a bootstrap loader from computer science. 
#
# So why would duplicating samples do us any good? If we do this multiple times and create a new classifier each time, then we will have our ensemble!

# + slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

titanic_df = pd.read_csv(
    f"{home}/csc-466-student/data/titanic.csv"
)
titanic_df.head()

# + slideshow={"slide_type": "subslide"}
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
titanic_df2 = titanic_df.loc[:,features]
titanic_df2['CabinLetter'] = titanic_df2['Cabin'].str.slice(0,1)
X = titanic_df2.drop('Cabin',axis=1)#.dropna()
X['CabinLetter'] = X['CabinLetter'].fillna("?")
X['Pclass'] = X['Pclass'].astype(str)
X['SibSp'] = X['SibSp'].astype(str)
X['Parch'] = X['Parch'].astype(str)
X = X.dropna()
X.head()
# -

t = titanic_df.loc[X.index,'Survived']

X2 = pd.get_dummies(X)

# + slideshow={"slide_type": "subslide"}
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.33, random_state=42)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,t_train)
y_test = clf.predict(X_test)
y_test

# + slideshow={"slide_type": "subslide"}
from sklearn.metrics import classification_report
print(classification_report(list(t_test), list(y_test)))

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** Write a function that creates N bootstrap samples. 
#
# Here is some code that creates a bootstrap sample.

# + slideshow={"slide_type": "fragment"}
from sklearn.utils import resample

X_train_sample, t_train_sample = resample(X_train, t_train)
t_train_sample

# + [markdown] slideshow={"slide_type": "subslide"}
# Can you beat a decision tree with more decision trees?

# + slideshow={"slide_type": "subslide"}
ntrees = 51
trees = []
# Your solution here

def vote(trees,X):
    votes = np.zeros((len(X),len(trees)))
    for i,tree in enumerate(trees):
        votes[:,i] = tree.predict(X)
    y = pd.DataFrame(votes,index=X.index).mode(axis=1).iloc[:,0].astype(int)
    return y

y_test = vote(trees,X_test)
print(classification_report(list(t_test), list(y_test)))

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Random Forest
#
# To produce more trees we need more randomness. A straightforward way to introduce randomness is to vary the features examined at each split. This is a random forest algorithm.
#
# * For each of N trees:
#     * create a new bootstrap sample of the training set
#     * use this bootstrap sample to train a decision tree
#     * at each node of the decision tree, randomly select m features, and compute the information gain (or Gini impurity) only on that set of features, selecting the optimal one
#     * repeat until the tree is complete

# + [markdown] slideshow={"slide_type": "subslide"}
# Let's see if we can do even better with random forest!

# + slideshow={"slide_type": "fragment"}
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X_train, t_train)
y_test = clf.predict(X_test)
print(classification_report(list(t_test), list(y_test)))

# + [markdown] slideshow={"slide_type": "subslide"}
# ## What other methods of ensemble learning is popular?
#
# Gradient boosting! This isn't talked about as much as deep learning, but it is the other big dog when it comes to online competitions on Kaggle. So what is it?
#
# Gradient boosting == Gradient descent and boosting
#
# The most straightforward gradient boosting algorithm is for regression, so we will start our discussion there.
#
# More information beyond our discussion can be found: http://www.chengli.io/tutorials/gradient_boosting.pdf

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Let’s play a game...
# You are given (x1, y1),(x2, y2), ...,(xn, yn), and the task is to fit a
# model F(x) to minimize square loss.
#
# You have a well meaning friend who gives you model F, but they aren't a perfect friend. A good friend, but not perfect. There are some mistakes: F(x1) = 0.8, while y1 = 0.9, and
# F(x2) = 1.4 while y2 = 1.3... 
#
# How can you improve this model?

# + [markdown] slideshow={"slide_type": "subslide"}
# Rule of the game:
# * You are not allowed to remove anything from F or change any parameter in F.
#
# **Stop and think:** What do you do?

# + [markdown] slideshow={"slide_type": "subslide"}
# You can add an additional model (regression tree) h to F, so the new prediction will be F(x) + h(x).
#
# What specifically, do you want to do?

# + [markdown] slideshow={"slide_type": "subslide"}
# Simple solution:
# * F(x1) + h(x1) = y1
# * F(x2) + h(x2) = y2
# * ...
# * F(xn) + h(xn) = yn

# + [markdown] slideshow={"slide_type": "subslide"}
# Or, equivalently, you could
# * h(x1) = y1 − F(x1)
# * h(x2) = y2 − F(x2)
# * ...
# * h(xn) = yn − F(xn)
#
# What have you done... You've created a new problem :)

# + [markdown] slideshow={"slide_type": "subslide"}
# yi − F(xi) are called residuals. These are the parts that existing
# model F cannot do well.
#
# The role of h is to compensate the shortcoming of existing model
# F.
#
# If you still aren't satisfied with the performance, you can do it again and again and again in a sequential manner. Don't forget one of our first images!

# + [markdown] slideshow={"slide_type": "subslide"}
#
# <img src="https://miro.medium.com/max/2000/1*zTgGBTQIMlASWm5QuS2UpA.jpeg">

# + [markdown] slideshow={"slide_type": "subslide"}
# And that's it! Time permitting this quarter, we will get into the theory of gradient boosting and where the gradient descent comes into play.
# -


