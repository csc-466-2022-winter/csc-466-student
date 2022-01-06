---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Lab 7

## Collaborative filtering and recommendations

### At the end of this lab, I should be able to
* Understand how item-item and user-user collaborative filtering perform recommendations
* Explain a experiment where we tested item-item versus user-user

**Note:** Exercises can be autograded and count towards your lab and assignment score. Problems are graded for participation.

```python
from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab7_helper
```

<!-- #region slideshow={"slide_type": "subslide"} -->
## Real dataset: Movielens

https://grouplens.org/datasets/movielens/

> MovieLens is a collaborative filtering system for movies. A
user of MovieLens rates movies using 1 to 5 stars, where 1 is "Awful" and 5 is "Must
See". MovieLens then uses the ratings of the community to recommend other movies
that user might be interested in, predict what that user might rate a movie,
or perform other tasks. - "Collaborative Filtering Recommender Systems"
<!-- #endregion -->

```python slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

ratings = pd.read_csv(f'{home}/csc-466-student/data/movielens-small/ratings.csv') # you might need to change this path
ratings = ratings.dropna()
ratings.head()
```

```python slideshow={"slide_type": "subslide"}
movies = pd.read_csv(f'{home}/csc-466-student/data/movielens-small/movies.csv')
movies = movies.dropna()
movies.head()
```

<!-- #region slideshow={"slide_type": "subslide"} -->
### Joining the data together
We need to join those two source dataframes into a single one called data. I do this by setting the index to movieId and then specifying an ``inner`` join which means that the movie has to exist on both sides of the join. Then I reset the index so that I can later set the multi-index of userId and movieId. The results of this are displayed below. Pandas is awesome, but it takes some getting used to how everything works.
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
data = movies.set_index('movieId').join(ratings.set_index('movieId'),how='inner').reset_index()
data = data.drop('timestamp',axis=1) # We won't need timestamp here
data.head()
```

```python
ratings = data.set_index(['userId','movieId'])['rating']
ratings # as Series
```

#### Exercise 1
I provide a structure for predicting recommentations using user-user collaborative filtering.  For this exercise, please complete the missing components.

``data_raw`` - your entire dataframe

``x_raw`` - the data from a single user

``N`` - neighborhood size

``frac`` - fraction for your test dataset

```python
mae = Lab7_helper.predict_user_user(ratings.unstack(),ratings.unstack().loc[1])
mae
```

#### Exercise 2
I provide a structure for predicting recommentations using item-item collaborative filtering. For this exercise, please complete the missing components.

```python
mae = Lab7_helper.predict_item_item(ratings.unstack(),ratings.unstack().loc[1])
mae
```

#### Problem 1
This is an open ended question that requires you to code. I have provided my own ratings for some of the movies in the dataset. What would you recommend to me based on my recommendations if you applied user-user filtering? Feel free to also change to your rankings. I ranked the top 5 movies according to the count of users who have ranked movies.

```python
data[['movieId','title']].value_counts()
```

```python
counts = data[['movieId','title']].value_counts().reset_index()
```

```python
user_ratings = pd.DataFrame(index=['Dr. Anderson'],columns=counts['title'])
user_ratings.loc["Dr. Anderson","Forrest Gump (1994)"] = 4
user_ratings.loc["Dr. Anderson","Shawshank Redemption, The (1994)"] = 5
user_ratings.loc["Dr. Anderson","Pulp Fiction (1994)"] = 3
user_ratings.loc["Dr. Anderson","Silence of the Lambs, The (1991)"] = 2
user_ratings.loc["Dr. Anderson","Matrix, The (1999)"] = 5
user_ratings
```

```python
ratings_reordered = ratings.unstack().T.loc[counts['movieId']].T # reorder the ratings to be the same as above
ratings_reordered.columns = user_ratings.columns
ratings_reordered
```

```python
### Your solution here
```

#### Problem 2
Repeat problem 1 but recommend movies using item-item. Any difference? Which one do you think is more reasonable?

```python
# Good job!
# Don't forget to push with ./submit.sh
```

<!-- #region -->
#### Having trouble with the test cases and the autograder?

You can always load up the answers for the autograder. The autograder runs your code and compares your answer to the expected answer. I manually review your code, so there is no need to hide this from you.

```python
import joblib
answers = joblib.load(f"{home}/csc-466-student/tests/answers_Lab7.joblib")
answers.keys()
```
<!-- #endregion -->

```python

```
