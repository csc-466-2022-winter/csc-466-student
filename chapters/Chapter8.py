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
# # Chapter 8 - Recommendation System
#
# Paul E. Anderson

# + slideshow={"slide_type": "skip"}
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system

# + [markdown] slideshow={"slide_type": "subslide"}
# Not every method falls directly under machine learning. Consider how you would recommend movies to a friend? You might ask them what kind of movies they like and then try to think of similar movies. And there we have it... you've performed _collaborative filtering_. You could go a step further and try to think of three other friends who have similar tastes to your friend. In collaborative filtering terminology you have defined the _neighborhood_.

# + [markdown] slideshow={"slide_type": "subslide"}
# User - any individual who provides ratings to a system
#
# Ratings:
# * Scalar ratings can consist of either numerical ratings, such as the 1-5 stars provided in MovieLens or ordinal ratings such as strongly agree, agree, neutral, disagree, strongly disagree.
# * Binary ratings model choices between agree/disagree or good/bad.
# * Unary ratings can indicate that a user has observed or purchased an item, or otherwise rated the item positively. The absence of a rating indicates that we have no information relating the user to the item (perhaps they purchased the item somewhere else).

# + [markdown] slideshow={"slide_type": "subslide"}
# Explicit ratings - users asked to provide an opinion on an item
#
# Implicit ratings - ratings are inferred from actions
#
# Examples: Users may visit a product page because they are interested in that product

# + [markdown] slideshow={"slide_type": "subslide"}
# What might someone do with CF:
# * Help me find new items I might like.
# * Advise me on a particular item. I have a particular item in mind; does the community know whether it is good or bad?
# * Help me find a user (or some users) I might like.
# * Help our group find something new that we might like.
# * Help me find an item, new or not.

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Real dataset: Movielens
#
# https://grouplens.org/datasets/movielens/
#
# > MovieLens is a collaborative filtering system for movies. A
# user of MovieLens rates movies using 1 to 5 stars, where 1 is "Awful" and 5 is "Must
# See". MovieLens then uses the ratings of the community to recommend other movies
# that user might be interested in, predict what that user might rate a movie,
# or perform other tasks. - "Collaborative Filtering Recommender Systems"

# + slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

ratings = pd.read_csv(f'{home}/csc-466-student/data/movielens-small/ratings.csv') # you might need to change this path
ratings = ratings.dropna()
ratings.head()

# + slideshow={"slide_type": "subslide"}
movies = pd.read_csv(f'{home}/csc-466-student/data/movielens-small/movies.csv')
movies = movies.dropna()
movies.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Joining the data together
# We need to join those two source dataframes into a single one called data. I do this by setting the index to movieId and then specifying an ``inner`` join which means that the movie has to exist on both sides of the join. Then I reset the index so that I can later set the multi-index of userId and movieId. The results of this are displayed below. Pandas is awesome, but it takes some getting used to how everything works.

# + slideshow={"slide_type": "subslide"}
data = movies.set_index('movieId').join(ratings.set_index('movieId'),how='inner').reset_index()
data = data.drop('timestamp',axis=1) # We won't need timestamp here
data.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# You can then calculate the average rating for a user:

# + slideshow={"slide_type": "fragment"}
data.set_index('userId').loc[1,'rating'].mean()

# + [markdown] slideshow={"slide_type": "subslide"}
# It would also be interesting to see the distribution of ratings across all users:

# + slideshow={"slide_type": "fragment"}
ax = data.reset_index().groupby('userId')['rating'].mean().plot.hist()
ax.set_xlabel('Rating');

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** How can we handle the genres column?

# + slideshow={"slide_type": "fragment"}
# Your solution here
genres.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** How would you calculate the average score per genre?

# + slideshow={"slide_type": "fragment"}
# Your solution here
means.sort_values()

# + [markdown] slideshow={"slide_type": "subslide"}
# How many unique users are in this small dataset?

# + slideshow={"slide_type": "fragment"}
len(ratings.userId.unique())

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Aside: Turning data into a matrix instead of a series
# The functions ``stack()`` and ``unstack()`` are called multiple times in this lab. These functions allow me to easily change from a dataframe to a series and back again. Below I'm changing from the Series object to a DataFrame. The important thing to note is that each row is now a user! NaN values are inserted where a user did not rate movie.

# + slideshow={"slide_type": "subslide"}
ratings = data.set_index(['userId','movieId'])['rating']
ratings # as Series

# + slideshow={"slide_type": "subslide"}
ratings.unstack()

# + slideshow={"slide_type": "subslide"}
ratings.unstack().stack() # we can do this all day

# + [markdown] slideshow={"slide_type": "subslide"}
# Remember our initial analogy. We really want a way to return the users who have similar taste in movies to our friend. This means that we need a way to measure distance between users. The first thing we need to consider is what happens when a user doesn't rate a movie? 

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** Why is it a bad idea to fill in 0 for missing values without any other modifications to the data? What could help?
#
# Your answer here

# + [markdown] slideshow={"slide_type": "subslide"}
# We've seen above that each user has a different baseline rating (average). It is a good idea to mean center each user:

# + slideshow={"slide_type": "subslide"}
ratings = ratings.unstack()
ratings = (ratings.T-ratings.mean(axis=1)).T
ratings = ratings.stack()
ratings

# + slideshow={"slide_type": "subslide"}
# Let's now convert this to a dataframe and fill in the zeros
ratings_df = ratings.unstack().fillna(0)
ratings_df

# + [markdown] slideshow={"slide_type": "subslide"}
# Let's say your friend is the first user:

# + slideshow={"slide_type": "fragment"}
x = ratings_df.loc[1]
x

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Finding neighborhood.
# If we are hoping to predict movies for this user, then user-user collaborative filtering says find the ``N`` users that are similar. We should definitely drop out user 1 because it makes no sense to recommend to yourself. We then compute the cosine similarity between this user ``x`` and all other users in the db. We then reverse sort them, and then display the results.
#
# **Stop and think:** Why would we not use Euclidean distance? Consider two users who fail to rate any movies?
#
# ##### Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# ${\displaystyle {\text{similarity}}=\cos(\theta )={\mathbf {A} \cdot \mathbf {B} \over \|\mathbf {A} \|\|\mathbf {B} \|}={\frac {\sum \limits _{i=1}^{n}{A_{i}B_{i}}}{{\sqrt {\sum \limits _{i=1}^{n}{A_{i}^{2}}}}{\sqrt {\sum \limits _{i=1}^{n}{B_{i}^{2}}}}}},}$

# + slideshow={"slide_type": "subslide"}
db = ratings_df.drop(1) # Make database of all users except user 1
x = ratings_df.loc[1]
sims = db.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)
sorted_sims = sims.sort_values()[::-1]
sorted_sims

# + slideshow={"slide_type": "subslide"}
sum(ratings_df.loc[53]!=0) # Just fyi

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Grabing similar users
# Let's set the network size to 10, and then grab those users :)

# + slideshow={"slide_type": "subslide"}
N=10
userIds = sorted_sims.dropna().iloc[:N].index
ratings_df.loc[userIds]

# + [markdown] slideshow={"slide_type": "subslide"}
# **Stop and think:** How do we make a prediction?
#
# Your solution here

# + slideshow={"slide_type": "subslide"}
movies_not_seen = x.loc[x != 0].index
movies_not_seen

# + slideshow={"slide_type": "fragment"}
average_ratings = db.loc[userIds].mean().loc[movies_not_seen].sort_values(ascending=False)
average_ratings
# -

data[['movieId','title']].drop_duplicates().set_index('movieId').loc[average_ratings.index]

# + [markdown] slideshow={"slide_type": "subslide"}
# ### What if we want to weight by the distance?
#
# In other words, if a user is closer to our query user, then they should count more.

# + slideshow={"slide_type": "fragment"}
average_ratings_weighted = (db.loc[userIds].multiply(sorted_sims.iloc[:N],axis=0).sum()/sorted_sims.iloc[:N].sum()).sort_values(ascending=False)
average_ratings_weighted

# + slideshow={"slide_type": "subslide"}
data[['movieId','title']].drop_duplicates().set_index('movieId').loc[average_ratings_weighted.index]

# + [markdown] slideshow={"slide_type": "subslide"}
# ## User-user small dataset example
# -

data2 = data.copy()
data = data['rating']

# + slideshow={"slide_type": "fragment"}
# grab some movies that were watched a lot
r=(ratings_df > 0).sum()
our_movies = r.sort_values(ascending=False).iloc[:10].index
our_movies

# + slideshow={"slide_type": "subslide"}
our_data = ratings_df[our_movies] # grab only those movies

# + slideshow={"slide_type": "subslide"}
# Now grab just the users
our_users = (our_data>0).sum(axis=1).sort_values(ascending=False).iloc[:10].index

# + slideshow={"slide_type": "subslide"}
train_data = our_data.loc[our_users]
train_data

# + [markdown] slideshow={"slide_type": "subslide"}
# It doesn't serve our purpose to have no missing values, so let's put some back in.

# + slideshow={"slide_type": "fragment"}
test_data = train_data.copy()
test_data.iloc[0,8] = np.NaN
test_data.iloc[1,8] = np.NaN
test_data.iloc[0,6] = np.NaN
test_data.iloc[5,8] = np.NaN
test_data.iloc[0,2] = np.NaN
test_data.iloc[3,8] = np.NaN
test_data.loc[352,593] = np.NaN
test_data.loc[352,527] = np.NaN

# + slideshow={"slide_type": "subslide"}
test_data

# + slideshow={"slide_type": "subslide"}
user_id = 307

# + slideshow={"slide_type": "subslide"}
test_data = (test_data.T-test_data.T.mean()).T # mean center everything
test_data.loc[user_id].mean() # check the mean of user 610
# -

x_raw = test_data.loc[user_id] # x_raw is a user
x_raw

data_raw = test_data.copy() # keep a copy of test_data that doesn't have any missing values
test_data = test_data.fillna(0) # fill in missing values

# we need to split this up into training and test sets
from sklearn.model_selection import train_test_split
train_movies, test_movies = train_test_split(x_raw.dropna(),test_size=0.2,random_state=1)
display(train_movies)
display(test_movies)

# but we just wanted the movies and not the ratings
train_movies, test_movies = train_test_split(x_raw.dropna().index,test_size=0.2,random_state=1)
print('Training movies')
display(train_movies)
print('Testing movies')
display(test_movies)

test_data

db = test_data.drop(x_raw.name) # remove this user
db

movie = test_movies[0] # pick a movie in our test set
display(movie)
display(db)
# We should remove any users that did not rate the movie we are interested in predicting. 
# How would including them help us?
db_subset = db.loc[np.isnan(data_raw.drop(x_raw.name)[movie])==False]
display(db_subset)

# In order to make the cosine similarity work, we need to have the same dimensions in db_subset and x
# But we want to make sure that the test movies are removed because well they are for testing purposes
x = x_raw.copy()
x.loc[test_movies] = np.NaN
x = x.fillna(0)
x

# Now we can actually compute the cosine similarity. This apply function is basically just a for loop over each user
sims = (db_subset.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)+1)/2

N = 2 # Set the neighborhood to 2 and select the users after sorting
sims.sort_values(ascending=False).iloc[:N]

# But we don't want the similarity scores, just the user ids
neighbors = sims.sort_values(ascending=False).iloc[:N].index
neighbors

# How did our neighborhood rank that movie?
test_data.loc[neighbors,movie]

# Finally! Here is our prediction (unweighted)
pred = test_data.loc[neighbors,movie].mean()
pred

# What about weighted?
top_sims = sims.sort_values(ascending=False).iloc[:N]
top_sims

# Here is our prediction with weighting
weighted_pred = test_data.loc[neighbors,movie].multiply(top_sims,axis=0).sum()/top_sims.sum()
weighted_pred

# How does this compare?
actual = x_raw.loc[movie]
actual


print("MAE of unweighted:",np.abs(actual-pred))
print("MAE of weighted:",np.abs(actual-weighted_pred))

# ## Item-item on the same small dataset
# Let's review what we have from above that becomes our input. Item - item works really similar to user-user but the information is now users ratings for each item instead of item ratings for each user (i.e., we transpo.

data_raw

# We are going to need to transform this
data_raw.T

x_raw

train_movies

test_movies

# This is the movie we are still trying to predict (i.e., from the testing set we pick the first one)
movie

# The intuition behind item-item is we want to predict the rating of a movie based on the user's ratings on similar movies. In other words, if we knew that most similar movies to 527 were 356, 318, and 296, then we would calculate our prediction like this:

# the use of 'rating' is just an artifact of pandas transformations
ids = [356,318,296]
x_raw.loc[ids]

# so we could predict like this
x_raw.loc[ids].mean()

# +
# but wait, why would we even includ movie 296? The above mean ignores this in the calculation,
# so it is better to just prevent this from happening, we can do that when we search the neighborhood!
# -

test_data = data_raw.T.fillna(0)
test_data

# The following lines are the same as the single line left below
x = test_data.loc[movie].drop(x_raw.name)
x

db_subset = test_data.loc[train_movies].drop(x_raw.name,axis=1)
db_subset

sims = (db_subset.apply(lambda y: (y.values*x.values).sum()/(np.sqrt((y**2).sum())*np.sqrt((x**2).sum())),axis=1)+1)/2
sims

top_sims = sims.sort_values(ascending=False).iloc[:N]
top_sims

ids = top_sims.index
ids

pred = x_raw.loc[ids].mean()
pred

x_raw.loc[ids]

top_sims

weighted_pred = x_raw.loc[ids].multiply(top_sims,axis=0).sum()/top_sims.sum()
weighted_pred

print("MAE of unweighted:",np.abs(actual-pred))
print("MAE of weighted:",np.abs(actual-weighted_pred))


