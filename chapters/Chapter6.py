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
# # Chapter 6 - Unsupervised Learning: Clustering
#
# Paul E. Anderson

# + slideshow={"slide_type": "skip"}
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system

# + [markdown] slideshow={"slide_type": "subslide"}
# In many applications, observations need to be divided into similar groups based on observed features. This is done at the beginning of many data projects. It is often exploratory in nature, and helps identify structure in your data. At other times, clustering/partitioning is the main objective. For example, retailers may want to divide potential customers into groups, in order to target a marketing campaign at the customers who are most likely to respond positively.

# + [markdown] slideshow={"slide_type": "subslide"}
# The general problem of grouping observations based on observed features is known as _clustering_. Where classification focuses on a matrix $X$ and a vector $y$ of labels, clustering ignores $y$ and focuses solely on $X$.
#
# This is the reason we call clustering an example of _unsupervised learning_. The supervision is contained in the $y$ vector, and we are removing that from direct analysis.

# + [markdown] slideshow={"slide_type": "subslide"}
# Here is an analogy from childhood. Two children are playing with blocks of different colors. One is accompanied by an adult who provides feedback about the color and shape of the blocks. This is supervised learning. The other child is observed but no feedback is provided. Both children play with the blocks, but the second child is doing so in what machine learning experts would say is unsupervised.
#
# <img src="https://github.com/dlsun/pods/blob/master/07-Unsupervised-Learning/shape_sorter.jpg?raw=1">

# + [markdown] slideshow={"slide_type": "subslide"}
# #### K-means algorithm
# $K$-means is an algorithm for finding clusters in data. The idea behind $k$-means is simple: each cluster has a "center" point called the **centroid**, and each observation is associated with the cluster of its nearest centroid. The challenge is finding those centroids. The $k$-means algorithm starts with a random guess for the centroids and iteratively improves them.

# + [markdown] slideshow={"slide_type": "subslide"}
# The steps are as follows:
#
# 1. Initialize $k$ centroids at random.
# 2. Assign each point to the cluster of its nearest centroid.
# 3. (After reassignment, each centroid may no longer be at the center of its cluster.) Recompute each centroid based on the points assigned to its cluster.
# 4. Repeat steps 2 and 3 until no points change clusters.

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Stop and think: Will this algorithm converge at all times?
#
# #### Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Stop and think: How do you pick the starting locations?
#
# #### Your solution here
# -

# #### Stop and think: What is the best way?
#
# #### Your solution here

# #### Stop and think: How do you measure a good solution in clustering?
#
# #### Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# ### K-means from scratch

# + slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

df = pd.read_csv(
    f"{home}/csc-466-student/data/breast_cancer_three_gene.csv",index_col=0
)
df.head()

# + slideshow={"slide_type": "subslide"}
X = df[['ESR1','AURKA']]#,'ERBB2']]
# Stop and think: What happens when I put in the third variable? Did your code work? What about the plots?
X.head()


# + slideshow={"slide_type": "subslide"}
# Stop and think: Implement a function that calculates the distance between two vectors using Euclidean distance
def distance(x,c):
    d = None
    return d

distance(X.loc[0],X.loc[1])

# + slideshow={"slide_type": "subslide"}
# Stop and think: How would you find k random means for starting the clustering?
means = None
k = 6

means

# +
import altair as alt

alt.Chart(X).mark_circle(size=60).encode(
    x='ESR1',
    y='AURKA') + \
alt.Chart(means).mark_circle(color='black',size=200).encode(
    x='ESR1',
    y='AURKA')

# + slideshow={"slide_type": "subslide"}
# Stop and think: How would you assign each datapoint to a mean?
# Stop and think: How would you compute the distortion?
clusters = []
distortion = 0
# Your solution here
Xc = X.copy()
Xc['cluster']=clusters
Xc.head()

# + slideshow={"slide_type": "subslide"}
distortion

# + slideshow={"slide_type": "subslide"}
# Stop and think: How would you recompute the mean?
means = None
# Your solution here
means

# + slideshow={"slide_type": "subslide"}
# Stop and think: Now put it all together and iterate
# Your solution here

# + slideshow={"slide_type": "subslide"}
# Stop and think: Now try it all again with different initial means. Did the distortion go up or down?
# Your interpretation here

# + slideshow={"slide_type": "subslide"}
# Stop and think: How would you visualize the clustering?
# Your solution here

# + slideshow={"slide_type": "subslide"}
# Stop and think: Can you overlay the actual subtypes on this in any manner?
# Your solution here
