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
# # Chapter 7 - Dimensionality Reduction
#
# Paul E. Anderson

# + slideshow={"slide_type": "skip"}
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system

# + [markdown] slideshow={"slide_type": "subslide"}
# We saw in the last chapter that it was difficult to visualize more than two dimensions. What are our options? Choose two dimensions to visualize at time? Plot three dimensions that are then projected down into two dimensions? What if you could create new dimensions that are "optimized" combinations of the original dimensions? 
#
# Finding/creating those new dimensions is what this chapter is all about. There are many different ways to create new dimensions, and we will focus on one of the most popular methods: Principal Component Analysis (PCA). 

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Example motivation: Visualizing three gene data

# + slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

df = pd.read_csv(
    f"{home}/csc-466-student/data/breast_cancer_three_gene.csv",index_col=0
)
df.head()

# + slideshow={"slide_type": "subslide"}
X = df[['ESR1','AURKA','ERBB2']]
X.head()

# + slideshow={"slide_type": "subslide"}
# Three ways to plot
import altair as alt

g1 = alt.Chart(X).mark_point().encode(
    x='ESR1',
    y='AURKA'
)

g2 = alt.Chart(X).mark_point().encode(
    x='ESR1',
    y='ERBB2'
)

g3 = alt.Chart(X).mark_point().encode(
    x='AURKA',
    y='ERBB2'
)

# + slideshow={"slide_type": "subslide"}
g1

# + slideshow={"slide_type": "subslide"}
g2

# + slideshow={"slide_type": "subslide"}
g3

# + [markdown] slideshow={"slide_type": "subslide"}
# It would be very nice to have all of this on a single plot? So we are left with wondering how can we do that in a way that minimizes the amount of information that we are not seeing?

# + slideshow={"slide_type": "subslide"}
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
Xnew = pd.DataFrame(pca.fit_transform(X),columns=["PC1","PC2","PC3"])
Xnew.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** What is this printing?

# + slideshow={"slide_type": "fragment"}
print(pca.explained_variance_ratio_)

# + [markdown] slideshow={"slide_type": "fragment"}
# **Your answer here**

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** What is this printing?

# + slideshow={"slide_type": "fragment"}
print(pca.singular_values_)

# + [markdown] slideshow={"slide_type": "fragment"}
# **Your answer here**

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** What is this printing?

# + slideshow={"slide_type": "fragment"}
print(pca.components_)

# + [markdown] slideshow={"slide_type": "fragment"}
# **Your answer here**

# + [markdown] slideshow={"slide_type": "subslide"}
# ### We can now visualize our data!

# + slideshow={"slide_type": "fragment"}
source = Xnew.copy()
source['Subtype'] = df['Subtype']

alt.Chart(source).mark_point().encode(
    x='PC1',
    y='PC2',
    color='Subtype'
)

# + [markdown] slideshow={"slide_type": "subslide"}
# ## PCA from scratch
# To understand how all of this is working, we will implement a PCA that goes from two dimensions to one. It will match sklearn's answers.
#
# We will reduce two dimensions into a single dimension. The work we do here, can easily be extended to more than two dimensions.

# + slideshow={"slide_type": "subslide"}
X = df[['ESR1','AURKA']] # we will reduce from two variables to one
X.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Run PCA and summarize

# + slideshow={"slide_type": "fragment"}
import numpy as np
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
Xnew = pd.DataFrame(pca.fit_transform(X),columns=["PC1","PC2"])
print('Components')
print(pca.components_)
print('Singular values')
print(pca.singular_values_)
print('Explained variance ratio')
print(pca.explained_variance_ratio_)

# + slideshow={"slide_type": "skip"}
source = Xnew.copy()
source['Subtype'] = df['Subtype']

g = alt.Chart(source).mark_point().encode(
    x='PC1',
    y='PC2',
    color='Subtype'
)

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Visualize

# + slideshow={"slide_type": "fragment"}
g

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** What is the covariance matrix?

# + slideshow={"slide_type": "fragment"}
COV = pd.DataFrame(columns=X.columns,index=X.columns)
COV
# Your solution here

# + slideshow={"slide_type": "subslide"}
# Cheat code:
X.cov()

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Recall from slides:
#
# $\lambda^2 - (a + d)\lambda + (ad - bc) = 0$
#
# where

# + slideshow={"slide_type": "fragment"}
a = COV.iloc[0,0]
b = COV.iloc[0,1]
c = COV.iloc[1,0]
d = COV.iloc[1,1]

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Find those roots!

# + slideshow={"slide_type": "fragment"}
import numpy as np
# See np.roots
## BEGIN SOLUTUION
coeff = [1,-(a+d),(a*d - b*c)]
lambdas = np.roots(coeff)
# Your solution here
lambdas

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Recall from slides:
#
# $(\Sigma - I*\lambda)\vec{v} = 0$
#
# If $\vec{v} = (x,y)$, then
#
# $(a-\lambda)*x + b*y = 0$ and $c*x + (d-\lambda)*y = 0$
#
# To solve either of these equations, let $x=1$ and then solve for $y$. Take that vector and normalize it to have a length of 1.

# + slideshow={"slide_type": "fragment"}
# Your solution here
y

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Normalize the vectors to have a length of 1

# + slideshow={"slide_type": "fragment"}
v1 = np.array([1,y[0]])
v1 = v1/np.sqrt(np.sum(v1**2))
v2 = np.array([1,y[1]])
v2 = v2/np.sqrt(np.sum(v2**2))
print("First eigen vector:",v1)
print("Second eigen vector:",v2)

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** Are these identical? If not, are they equivalent?
#
# Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** How do you project a sample into the new first dimension?

# + slideshow={"slide_type": "subslide"}
## BEGIN SOLTUION
PC1_score = np.sum(X.iloc[0]*v1)
# Your solution here
PC1_score

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Cheat code

# + slideshow={"slide_type": "fragment"}
PC1 = np.dot(X,v1)
PC1

# + slideshow={"slide_type": "subslide"}
PC2 = np.dot(X,v2)
PC2

# + slideshow={"slide_type": "skip"}
source = pd.DataFrame({"PC1":PC1,"PC2":PC2,"Subtype":df["Subtype"]})

g = alt.Chart(source).mark_point().encode(
    x='PC1',
    y='PC2',
    color='Subtype'
)

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Plot the results!
#
# **Stop and think** Does this look the same as above?
#
# Your answer here

# + slideshow={"slide_type": "fragment"}
g

# + [markdown] slideshow={"slide_type": "subslide"}
# ## What about as a preprocessing step?

# + slideshow={"slide_type": "subslide"}
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import numpy as np
import sklearn

X = df[['ESR1','AURKA','ERBB2']]
t = pd.get_dummies(df['Subtype'])

from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.33, random_state=42)

mlp = MLPClassifier()
mlp.fit(X_train, t_train)

y_test = mlp.predict(X_test)

# + slideshow={"slide_type": "subslide"}
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
import numpy as np
import sklearn

X = df[['ESR1','AURKA','ERBB2']]
t = pd.get_dummies(df['Subtype'])

pca = PCA(n_components=3)
Xnew = pd.DataFrame(pca.fit_transform(X),columns=["PC1","PC2","PC3"])

from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(Xnew, t, test_size=0.33, random_state=42)

mlp = MLPClassifier()
mlp.fit(X_train, t_train)

y_test_pca = mlp.predict(X_test)

# + slideshow={"slide_type": "subslide"}
print(sklearn.metrics.classification_report(t_test,y_test))

# + slideshow={"slide_type": "subslide"}
print(sklearn.metrics.classification_report(t_test,y_test_pca))
