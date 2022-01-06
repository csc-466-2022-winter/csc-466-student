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
# # Chapter 4 - Decision Trees
#
# Paul E. Anderson

# + slideshow={"slide_type": "skip"}
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system

# + [markdown] slideshow={"slide_type": "subslide"}
# ## How would you communicate your decision about a typical Thursday night?
#
# Let's all pretend this isn't a pandemic for a second, and you have a new roommate who is part of an exchange program from your favorite foreign city. They are a little confused about what to do on their first Thursday night in town, so they ask you for help. You talk to them for a while, but it becomes clear they really won't leave you alone until you diagram your decision making progress. Let's take 5-10 minutes and come up with a tree you can share with the class :)
#
# Here are your features:
# * Deadline? $\in$ {Urgent, Near, None}
# * Lazy? $\in$ {Yes, No}
# * Party? $\in$ {Yes, No party going on tonight}
#
# You are trying to predict one of the following activities:
# * Activity $\in$ {Party, TV, Pub, Study}

# + [markdown] slideshow={"slide_type": "subslide"}
# ### What if we needed a way to create a tree that generalized to many people?
#
# Onto our decision tree algorithm!

# + slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

df = pd.read_csv(f'{home}/csc-466-student/data/activity.csv')
df

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Let's play a game of 20 questions
#
# You've got 20 questions and you are trying to quess what your opponent is thinking by asking questions. 
#
# For each question, you are trying to ask something that gives you the most information given what you already know. 
#
# It is common to ask things like "Is it an animal?" before asking "Is it a cat?"
#
# We need a way to encode this mathematically, and this leads us to information theory.

# + [markdown] slideshow={"slide_type": "subslide"}
# We will reserve the writing of the decision tree algorithm to lab this week. Instead, we will focus on defining the process mathematically. To do this, we need to start discussing sets and metrics we can use about sets.
# -

# ### Entropy
#
# We will first discuss the entropy of the universe. If said universe consisted of a set of items. 
#
# Let $S$ be a set of items. Each element of $S$ can take on one of $Y$ values. We will write the unique values of $Y$ as $V_Y$.
#
# $p(y)$ proportion of the number of elements in class ${\displaystyle y}$ to the number of elements in set
#
# ${\displaystyle \mathrm {H} {(S)}=\sum _{y\in V_Y}{-p(y)\log _{2}p(y)}}$

# + [markdown] slideshow={"slide_type": "subslide"}
# Let's look at our concrete example:

# + slideshow={"slide_type": "fragment"}
S = df['Activity'].values
S

# + slideshow={"slide_type": "fragment"}
Y = df['Activity'].unique()
Y

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** How do you factor in prior knowledge (i.e., answers to prior questions)?

# + [markdown] slideshow={"slide_type": "fragment"}
# Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** How do you measure the entropy if you knew all the possible answers?

# + [markdown] slideshow={"slide_type": "fragment"}
# Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Now onto the ID3 algorithm!
# From Marsland Chapter 12:
#
#
# * If all examples have the same label:
#     * return a leaf with that label
# * Else if there are no features left to test:
#     * return a leaf with the most common label
# * Else:
#     * choose the feature $F$ that maximises the information gain of $S$ to be the next node using Equation (12.2) ˆ
#     * add a branch from the node for each possible value f in $F$
#     * for each branch:
#         * calculate $S_f$ by removing $F$ from the set of features
#         * recursively call the algorithm with $S_f$ , to compute the gain relative to the current set of examples

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Let's make our tree!
#
# To the whiteboard!

# + [markdown] slideshow={"slide_type": "subslide"}
# Now that we have a basic algorithm, what are some of the advances:
# * Continuous variables
# * C4.5 with pruning

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** How would you adapt for continuous variables?

# + [markdown] slideshow={"slide_type": "subslide"}
# So our next natural question is ... how do we compare algorithms?
#
# ##### Your solution here
# -


