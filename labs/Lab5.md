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

# Lab 5

## Ensemble Learning

For this lab, we will implement a bagging and boosting. Due to the increased simplicity when implementing regression with boosted trees over classification, we will stick to regression for this lab. Classification is a straightforward extension.

There are two main exercises and questions in the lab. 

**Note:** Exercises can be autograded and count towards your lab and assignment score. Problems are graded for participation.

```python
from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab5_helper
```

For developing this lab, we can our three gene dataset. We will try to predict the value of ESR1 from the other two genes.

```python
import pandas as pd
import numpy as np

df = pd.read_csv(
    f"{home}/csc-466-student/data/breast_cancer_three_gene.csv",index_col=0
)
df.head()
```

We need to do some simple preprocessing before our neural network can deal with this data. 

```python
X = df.drop('Subtype',axis=1)#.dropna()
X
```

```python
t = X['ESR1']
X2 = pd.get_dummies(X.drop('ESR1',axis=1))
X2
```

#### Exercise 1
Implement bagging using regression trees as the (weak) individual learner. You should be able to reach a similar accuray to my implementation.

```python
import numpy as np
learner = Lab5_helper.get_learner(X2,t) # Here is how to get a single weak learner to help build your ensembles
y = learner.predict(X2) # As usual, here is how to get predictions
RMSE = np.sqrt(((y-t)**2).sum()/len(t)) # Here is a sample calculation of the root mean squared error
print('Our prediction for Fare is off by',RMSE)
```

```python
from sklearn.model_selection import train_test_split

ntrials = 50
# when you are debugging you will want to lower this number
RMSEs = []
for trial in range(ntrials):
    X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.25,random_state=trial)
    trees = Lab5_helper.make_trees(X_train,t_train,ntrees=100)
    y = Lab5_helper.make_prediction(trees,X_test)
    RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
np.median(RMSEs)
```

**Problem 1:** How do we know this is behaving as expected when there is so much randomness? This is a very similar to a question about how can we compare two algorithms? Let's examine this by changing the number of trees in bagging between 25 and 100. Then we will see if we can detect a difference.

**Your answer here**

```python
results = pd.DataFrame({'RMSE':RMSEs,'Method':'Bagging (ntrees=100)'})
RMSEs = []
for trial in range(ntrials):
    X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.25,random_state=trial)
    trees = Lab5_helper.make_trees(X_train,t_train,ntrees=25)
    y = Lab5_helper.make_prediction(trees,X_test)
    RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
results2 = pd.DataFrame({'RMSE':RMSEs,'Method':'Bagging (ntrees=25)'})
results = results.append(results2)
```

```python
results.groupby('Method')['RMSE'].median()
```

```python
results.groupby('Method')['RMSE'].mean()
```

This looks promising! But what do the statistics tell us?

```python
import altair as alt

alt.Chart(results).mark_boxplot().encode(
    alt.Y("RMSE:Q"),
    x='Method',
).properties(width=300)
```

For a moment, let's not worry that the underlying distributions do not look normal. We know how to compare the average of two distributions, the t-test. Let's see what that says:

```python
pivot_results = results.pivot(columns='Method')
pivot_results.head()
```

```python
pivot_results.columns
```

```python
from scipy import stats
stats.stats.ttest_ind(pivot_results[('RMSE', 'Bagging (ntrees=100)')],pivot_results[('RMSE', 'Bagging (ntrees=25)')],equal_var = True)
```

So according to this test, we do not see any significant difference between the results.


**Problem 2.** What if you only used 2 trees? Is there a significant difference between 2 and 25 trees?

**Your answer here**

```python
RMSEs = []
for trial in range(ntrials):
    X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.25,random_state=trial)
    trees = Lab5_helper.make_trees(X_train,t_train,ntrees=2)
    y = Lab5_helper.make_prediction(trees,X_test)
    RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
results2 = pd.DataFrame({'RMSE':RMSEs,'Method':'Bagging (ntrees=2)'})
results = results.append(results2)
```

```python
results.groupby('Method')['RMSE'].mean()
```

```python
pivot_results = results.pivot(columns='Method')
pivot_results.head()
```

```python
stats.stats.ttest_ind(pivot_results[('RMSE', 'Bagging (ntrees=100)')],pivot_results[('RMSE', 'Bagging (ntrees=2)')],equal_var = True)
```

Finally, we can report a p-value < 0.05! Even if we were to discuss multiple test correction at this time, this would still be a significant result. So bagging is helping us!


#### Exercise 2
Implement boosting using regression trees as the (weak) individual learner. You should be able to reach a similar accuray to my implementation.

```python
RMSEs = []
for trial in range(ntrials):
    X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.25,random_state=trial)
    X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.25,random_state=trial)
    trees,train_RMSEs,val_RMSEs = Lab5_helper.make_trees_boost(X_train2, X_val, t_train2, t_val, max_ntrees=100)
    trees = Lab5_helper.cut_trees(trees,val_RMSEs)
    y = Lab5_helper.make_prediction_boost(trees,X_test)
    RMSEs.append(np.sqrt(((y-t_test)**2).sum()/len(t_test)))
results2 = pd.DataFrame({'RMSE':RMSEs,'Method':'Boosting (max_ntrees=100)'})
results = results.append(results2)
```

```python
source = pd.DataFrame({"train_RMSE":train_RMSEs,"val_RMSE":val_RMSEs, "ntrees":np.arange(len(val_RMSEs))}).melt(id_vars=['ntrees'])

alt.Chart(source).mark_line().encode(
    y = alt.Y("value:Q", scale=alt.Scale(domain=[0.1, 0.3])),
    x = 'ntrees',
    color='variable'
).properties(width=500)
```

```python
results.groupby('Method')['RMSE'].median().sort_values()
```

```python
results.groupby('Method')['RMSE'].mean().sort_values()
```

```python
np.mean(RMSEs)
```

```python
import altair as alt

alt.Chart(results).mark_boxplot().encode(
    alt.Y("RMSE:Q", scale=alt.Scale(domain=[0.2, 0.3])),
    x='Method',
).properties(width=300)
```

**Problem 3.** How would you compare (using t-test) whether the boosting algorithm is better than bagging (ntrees=100)?

```python
# Your solution here
```

```python
# Good job!
# Don't forget to push with ./submit.sh
```

<!-- #region -->
#### Having trouble with the test cases and the autograder?

You can always load up the answers for the autograder. The autograder runs your code and compares your answer to the expected answer. I manually review your code, so there is no need to hide this from you.

```python
import joblib
answers = joblib.load(f"{home}/csc-466-student/tests/answers_Lab5.joblib")
answers.keys()
```
<!-- #endregion -->
