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

# Ethics with an introduction to NumPy and Pandas on the side


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.

Please see the README for instructions on how to submit and obtain the lab.

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab1_helper

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

# Python, NumPy, and Pandas

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy and Pandas that we will rely upon routinely. 


## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)


## Exercises 1-5
For the following exercises please read the Python appendix in the Marsland textbook and answer problems below in the space provided and in Lab1_helper.py.


#### Exercise 1. Make an array a of size 6 × 4 where every element is a 2.

```python
a = Lab1_helper.exercise_1()
a
```

#### Exercise 2. Make an array b of size 6 × 4 that has 3 on the leading diagonal and 1 everywhere else. (You can do this without loops.)

```python
b = Lab1_helper.exercise_2()
b
```

#### Stop and think: Why can you multiply these two matrices together? Why does a * b work, but not dot(a,b)?

```python
a*b
```

```python
import numpy as np

np.dot(a,b)
```

#### YOUR SOLUTION HERE


#### Stop and Think: Compute dot(a.transpose(),b) and dot(a,b.transpose()). Why are the results different shapes?

```python
np.dot(a.transpose(),b)
```

```python
np.dot(a,b.transpose())
```

#### YOUR SOLUTION HERE


#### Exercise 3. Find the overall mean of matrix ``c``.

```python
np.random.seed(1)
c = np.random.rand(6,4)
display(c)
m = Lab1_helper.exercise_3(c)
m
```

#### Exercise 4. Find the column and row means of matrix ``c``.

```python
row_means,col_means = Lab1_helper.exercise_4(c)
display(row_means)
display(col_means)
```

#### Exercise 5. Write a function that consists of a set of loops that run through an array and counts the number of ones in it. Do the same thing without using a for loop. For inspiration, check out the following. You don't need to use all of them, but pick one.

```python
b == 1
```

```python
np.where(b == 1)
```

```python
np.array(b.flat)
```

```python
np.where(b.flat == 1)
```

```python
c1,c2 = Lab1_helper.exercise_5(b)
c1,c2
```

## Excercises 6-7
We will use Pandas at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html) before proceeding to any of the exercises below.


#### Exercise 6. Repeat Exercise 1, but create a Pandas DataFrame instead of a NumPy array.

```python
a = Lab1_helper.exercise_6()
a
```

#### Exercise 7. Repeat exercise 2 using a DataFrame instead.

```python
b = Lab1_helper.exercise_7()
b
```

#### Stop and think: What if we want to go from a pandas dataframe to a numpy array?

```python
b.values
```

## Exercises 8-10
Now let's look at a real dataset, and talk about ``.iloc`` and ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
import pandas as pd
titanic_df = pd.read_csv(
    f"{home}/csc-466-student/data/titanic.csv"
)
titanic_df
```

```python
titanic_df.index # default index
```

#### Stop and think: How would you set the index of the data frame to the Sex column and then select only those passengers who are female?

```python
df = titanic_df.set_index('Sex').loc['female']
df
```

What is important about the above statement is we have used ``.loc`` when we want to reference the pandas dataframe index. This index can be integers (starting at 0 or 1 or random). Further, it could just be a string like the example above. If you want a traditional index like you would in an array, then use ``iloc``.


#### Stop and think: Without setting or using the index, return a new dataframe with only those passengers who survived

```python
inxs = np.where(titanic_df.Survived==1)
titanic_df.iloc[inxs]
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


#### Exercise 8. Select the ``Name`` column without using .iloc?

```python
sel = Lab1_helper.exercise_8(titanic_df)
sel
```

#### Exercise 9. After setting the index to ``Sex``, select all passengers that are ``Male``?

```python
sel = Lab1_helper.exercise_9(titanic_df)
sel
```

<!-- #region hideOutput=false hideCode=true -->
#### Exercise 10. Reset the index of ``titanic_df_copy``.
<!-- #endregion -->

```python
titanic_df_copy = titanic_df.set_index('Name')
Lab1_helper.exercise_10(titanic_df_copy)
titanic_df_copy
```

# Ethics

We are finally ready to think about KDD ethics! 

We have preprocessed a dataset on loan applications to make this example appropriate for linear regression. The independent variable data is real and has not been modified apart from being transformed (e.g., Married=Yes => Married=1.). In other words, this is a real dataset with minimal modifcations. 

Our client is a loan company, they would like you to look at this historical data of 296 loans which have been approved for varying amounts and stored in the column LoanAmountApproved. They are interested in extracting which independent variables are the most influential/important when predicting the amount of the approved loan. Upon ethical review, they have determined that ``Gender`` is a protected column and should not be considered in the analysis.

```python
import pandas as pd

# Read in the data into a pandas dataframe
credit = pd.read_csv(f"{home}/csc-466-student/data/credit.csv",index_col=0)
credit.head()
```

#### Exercise 11. Construct a linear model model.

Your model should predict LoanAmountApproved using all of the columns except ``Gender`` which after an ethical review was deemed inappropriate to consider when make a determination on the amount of loan approved for an applicant.

Use ``sklearn.linear_model.LinearRegression`` with the default constructor arguments. You'll need to call ``.fit``. The documentation for this is so easy to get I almost hesitate to put a link in here, but here you go :)

https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares

```python
X = credit.drop(['Gender','LoanAmountApproved'],axis=1)
y = credit['LoanAmountApproved']

model = Lab1_helper.exercise_11(X,y)

# Here is code that takes the numpy array of coefficients stored in model.coef_ and formats it nicely for the screen
coef = pd.Series(model.coef_,index=X.columns)
coef.abs().sort_values(ascending=False) # this takes the absolute value and then sorts the values in descending order
```

Now let's write some code that calculates the mean absolute error of our model which is one measure of how good our model is performing. Looks like we are approximately $27K off in our model on average.

```python
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y,model.predict(X))
```

The company asks you for your interpretation of the model. You say that being married is a high indicator of receiving a high amount for a loan. This surprises some of your colleagues, but they think this is reasonable to them. Everyone seems happy with the work. However, an experienced data scientist on your team suggests you run a correlation of the columns used in the regression against the column ``Gender`` since it is considered a protected column. You do so quickly to satisfy this request and get:

```python
Xgender = X.copy()
Xgender['Gender'] = credit['Gender']
Xgender.corr().loc['Gender'].abs().sort_values(ascending=False).drop('Gender')
```

**The problems labeled "Problem X" will not be autograded.** They will be factored into your participation score at the end of the quarter. In other words, complete them :)


#### Problem 1: 

What do you think about the results? Specifically, is the fact that Married is correlated with Gender at a correlation of 0.36 concerning from an ethical standpoint? What do you as an individual think? Can you think of any suggestions about what to do? What ethical framework did you use in your decision making process: Utilitarianism, Deontology, or Virtue-ethics.


Let's assume your suggestion was to remove it. Do so and then compare the accuracy of the model

```python
from sklearn.linear_model import LinearRegression

# YOUR SOLUTION HERE
# X2 = # make this a dataframe without the married column
# model2 = # make this your linear model
mean_absolute_error(y,model2.predict(X2))
```

#### Problem 2:

What do you think now, should you drop it? Your prediction is now off by more than $10,000. Is this ok?

#### YOUR SOLUTION HERE

```python
# Good job!
# Don't forget to push with ./submit.sh
```

```python

```

```python

```

```python

```

```python

```
