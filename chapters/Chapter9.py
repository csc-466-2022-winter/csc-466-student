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
# # Chapter 9 - Text Analysis and Natural Language Processing
#
# Paul E. Anderson

# + slideshow={"slide_type": "skip"}
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system

# + [markdown] slideshow={"slide_type": "subslide"}
# In order for any of our previous methods to work on textual data, we must convert natural language fragments, paragraphs, sentences, books, blogs, etc into numeric data. We will introduce several methods to do this and apply them to real data. I have a soft spot for an older dating site called OK Cupid. I met my wife on OK cupid :)

# + slideshow={"slide_type": "subslide"}
import pandas as pd
df = pd.read_csv(f'{home}/csc-466-student/data/okcupid.csv')
df.head()
# -

# Some text data is categorical, and therefore, processing it is relatively straightforward.

# + slideshow={"slide_type": "subslide"}
df['education'].value_counts().plot.bar();

# + [markdown] slideshow={"slide_type": "subslide"}
# But what about the essay?
# -

text = df['essay9'].fillna("").str.replace('<[^<]+?>', '')

# + slideshow={"slide_type": "fragment"}
text

# + [markdown] slideshow={"slide_type": "subslide"}
# How might you we make a recommendation based on the essay?

# + slideshow={"slide_type": "subslide"}
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(text.fillna("").head())
X

# + [markdown] slideshow={"slide_type": "subslide"}
# Well that is not helpful! Notice that CountVectorizer returns the term-frequency matrix, not as a DataFrame or even as a numpy array, but as a scipy sparse matrix. A sparse matrix is one whose entries are mostly zeroes. Instead of storing individual values, we can simply store the locations of the non-zero entries and their values. This representation offers substantial memory savings because most of the elements are zero and thus not stored. 
#
# But what if you want to look at a subset of the data? How do we convert a sparse matrix to dense and label it? Careful doing this on a large dataset.

# + slideshow={"slide_type": "subslide"}
Xdense = pd.DataFrame(X.todense(),columns=vectorizer.get_feature_names())
Xdense

# + [markdown] slideshow={"slide_type": "subslide"}
# This is a tabular representation of what is called a bag of words. A bag of words reduces a document to the multiset of its words, ignoring grammar and word order. (A multiset is like a set, except that elements are allowed to appear more than once.)
#
# So, for example, the bag of words representation of "I am Sam. Sam I am." (the first two lines of Green Eggs and Ham) would be {I, I, am, am, Sam, Sam}. In Python, an easy way to represent multisets is with dictionaries, where the keys are the (unique) words and the values are the counts. So we would represent the above bag of words as {"I": 2, "am": 2, "Sam": 2}. However, such a representation is not conducive to recommendations.

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** What word appears most often?

# + slideshow={"slide_type": "fragment"}
# Your solution here
sums.sort_values(ascending=False).head(30)

# + slideshow={"slide_type": "fragment"}
sums.sort_values(ascending=False).iloc[:20].plot.bar();

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** Is the fact that _and_ is the most common word interesting?
#
# ###### Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# ### TF-IDF: Term frequency inverse document frequency
# The problem with term frequencies (TF) is that common words like "the" and "that" tend to have high counts and dominate. 
#
# A better indicator of whether two documents are similar is if they share rare words. 

# + [markdown] slideshow={"slide_type": "subslide"}
# In TF-IDF we take term frequency and re-weight each term by how many documents that term appears in (i.e., the document frequency). 
#
# We want words that appear in fewer documents to get more weight, so we take the inverse document frequency (IDF). 
#
# We take the logarithm of IDF because the distribution of IDFs is heavily skewed to the right. So in the end, the formula for IDF is:
#
# $$ \textrm{idf}(t, D) = \log \frac{\text{# of documents}}{\text{# of documents containing $t$}} = \log \frac{|D|}{|d \in D: t \in d|}. $$
# (Sometimes, $1$ will be added to the denominator to prevent division by zero, if there are terms in the vocabulary that do not appear in the corpus.)
#
# To calculate TF-IDF, we simply multiply the term frequencies by the inverse document frequencies:
#
# $$ \textrm{tf-idf}(d, t, D) = \textrm{tf}(d, t) \cdot \textrm{idf}(t, D). $$
# Notice that unlike TF, the TF-IDF representation of a given document depends on the entire corpus of documents.

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** Let's see how our new vectorization changes are word ranking:

# + slideshow={"slide_type": "fragment"}
from sklearn.feature_extraction.text import TfidfVectorizer
# Your solution here
pd.Series(vec.idf_,index=vec.get_feature_names()).sort_values()

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** Should we use the cosine similarity or Euclidean distance metrics?

# + slideshow={"slide_type": "subslide"}
from sklearn.metrics.pairwise import cosine_similarity

sims = pd.DataFrame(cosine_similarity(tf_idf_sparse))
sims
# -

import numpy as np
sims.values[np.tril_indices(len(sims))] = np.NaN
sims

sims.stack().sort_values(ascending=False)

top_ix = sims.stack().sort_values(ascending=False).index[0]

top_ix

# + slideshow={"slide_type": "subslide"}
df.loc[list(top_ix)]
# -

list(text.loc[list(top_ix)])

# **Stop and think:** Let's go a little farther down the list and see what other similar essays you can find.

# +
# Your solution here
# -

# **Stop and think:** Convert the self-summary variable (essay0) in the OKCupid data set to a TF-IDF representation. Use this to find a match for user 61 based on what he says he is looking for in a partner (essay9).


