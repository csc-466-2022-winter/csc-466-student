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

# Lab 2

## Bayesian Classifier and Feature Importance

**Note:** Exercises can be autograded and count towards your lab and assignment score. Problems are graded for participation.

```python
from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab2_helper
```

For this lab, we are going to first implement an empirical naive bayesian classifier, then implement feature importance measures and apply it to a dataset, and finally, we will examine the affect of modifying the priors.

For developing this lab, we can use famous Titanic Kaggle dataset. Description of the data is found https://www.kaggle.com/c/titanic/data.

```python
import pandas as pd
titanic_df = pd.read_csv(
    f"{home}/csc-466-student/data/titanic.csv"
)
titanic_df.head()
```

We only need a few columns, and I will also perform some preprocessing for you:

```python
features = ['Pclass','Survived','Sex','Age']
titanic_df = titanic_df.loc[:,features]
print('Before')
display(titanic_df.head())
titanic_df.loc[:,'Pclass']=titanic_df['Pclass'].fillna(titanic_df['Pclass'].mode()).astype(int)
titanic_df.loc[:,'Age']=titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df.loc[:,'Age']=(titanic_df['Age']/10).astype(str).str[0].astype(int)*10
titranic_df = titanic_df.dropna()
print('After')
titanic_df.head()
```

```python
titanic_df.describe()
```

#### Problem 1
In your own words, describe the preprocessing steps I took above.


Your solution here.


#### Exercise 1
Create a function to determine the prior probability of ALL the classes in ``y``. The result must be in the form of a Python dictionary such as ``priors = {'Survived=0': 0.4, 'Survived=1': 0.6}``.

```python
survived_priors = Lab2_helper.compute_priors(titanic_df['Survived'])
survived_priors
```

```python
Lab2_helper.compute_priors(titanic_df['Age'])
```

```python
y_example = titanic_df['Age']
y_example.name
```

#### Exercise 2
Create a function to calculate the specific class conditional probability. Assume x and y are pd.Series objects. Assume xv and yv are specific values. This function should return $\Pr(x==xv|y==yv)$.

```python
prob = Lab2_helper.specific_class_conditional(titanic_df['Sex'],'female',titanic_df['Survived'],0)
prob
```

#### Exercise 3
Now construct a dictionary based data structure that stores all possible class conditional probabilities (e.g., loop through all possible combinations of values). The keys in your dictionary should be of the form "pclass=1|survived=0". ``X`` is a ``pd.DataFrame`` object and ``y`` is a ``pd.Series`` object. You can retrieve the name of the series object ``y`` by accessing ``y.name``.

Aside: I know it might be a bit annoying to store the key of this dictionary as a string instead of as say a tuple of tuples, but I think the way this prints for folks learning this is reason enough to use strings in this instance.

```python
X = titanic_df.drop("Survived",axis=1)
y = titanic_df["Survived"]
probs = Lab2_helper.class_conditional(X,y)
probs
```

#### Exercise 4
Now you are ready to calculate the posterior probabilities for a given sample. Write and test the following function that returns a dictionary where the keys are of the form "Survived=0|Pclass=1,Sex=male,Age=60". Make sure you return 0.5 if the specific combination of values does not exist. ``probs`` and ``priors`` are defined the same as above. ``x`` is a pd.Series object that represents a specific example/sample from our dataset.

```python
probs = Lab2_helper.class_conditional(X,y)
priors = Lab2_helper.compute_priors(y)
x = titanic_df.drop("Survived",axis=1).loc[2]
x
```

```python
post_probs = Lab2_helper.posteriors(probs,priors,x)
post_probs
```

#### Exercise 5
All this is great, but how would you evaluate how we are doing? Create a function called train_test_split that splits our dataframe into a training and testing dataset. To make sure this is done randomly, I have inserted a shuffle into the code for you. The ``test_frac`` is the fraction of the dataset that will be held out for testing.

```python
import numpy as np
np.random.seed(2)
Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
print('Xtrain')
display(Xtrain.head())
print('ytrain')
display(ytrain.head())
print('Xtest')
display(Xtest.head())
print('ytest')
display(ytest.head())
```

#### Exercise 6
For this exercise, create a training dataset of size 50% and then using your solutions to previous exercises, find the prediction accuracy for the test dataset. 

```python
ytest
```

```python
np.random.seed(0)
Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
accuracy = Lab2_helper.exercise_6(Xtrain,ytrain,Xtest,ytest)
accuracy
```

That's not bad!

#### Problem 2:
Is that better than guessing all passengers died? What is the accuracy on the test set if you guessed all passengers died?

Your answer here


**Before proceeding**, make sure you have read the required reading of section 5.5.1 of [this book by Christoph Molnar](https://christophm.github.io/interpretable-ml-book/feature-importance.html). 

#### Excercise 7. Create a function that returns the test based feature importance for our Bayesian classifier.

```python
np.random.seed(0)
Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
importances = Lab2_helper.exercise_7(Xtrain,ytrain,Xtest,ytest)
importances
```

#### Excercise 8. Create a function that returns the train based feature importance for our Bayesian classifier.

```python
np.random.seed(0)
Xtrain,ytrain,Xtest,ytest=Lab2_helper.train_test_split(X,y)
importances = Lab2_helper.exercise_8(Xtrain,ytrain,Xtest,ytest)
importances
```

#### Problem 3. After you implement this, what is the most important feature? 

Your answer here


#### Problem 4: What are the differences between the two sets of importances?

Your answer here


#### Problem 5: What does a negative value for the importances mean? Consider that it is a very small importance, so what does it say about these features? Consider that question in the context of what you are permuting (training or testing).

Your answer here


Another thing to add to your brain, are there any correlations between features?

```python
Xtrain.corr()
```

```python
Xtest.corr()
```

```python
# Good job!
# Don't forget to push with ./submit.sh
```

#### Having trouble with the test cases and the autograder?

You can always load up the answers for the autograder. The autograder runs your code and compares your answer to the expected answer. I manually review your code, so there is no need to hide this from you.

```python
import joblib
answers = joblib.load(f"{home}/csc-466-student/tests/answers_Lab2.joblib")
answers.keys()
```

```python
answers['exercise_8']
```

```python

```

```python

```

```python

```
