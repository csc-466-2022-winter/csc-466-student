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

# Lab 3

## Single Layer Neural Network, Evaluation, and Interpretation

This lab is designed to teach you about different strategies for evaluating neural networks and interpreting the results. For this lab, we are going to study different methods for intrepreting a neural network. We are going to use a modified version of the perceptron learning algorithm. I have modified it to use the sigmoid activation function. This means our learning rule is updated. 

**Note:** Exercises can be autograded and count towards your lab and assignment score. Problems are graded for participation.

```python
from pathlib import Path 
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab3_helper
```

For developing this lab, we can use famous Titanic Kaggle dataset. Description of the data is found https://www.kaggle.com/c/titanic/data.

```python
import pandas as pd
titanic_df = pd.read_csv(
    f"{home}/csc-466-student/data/titanic.csv"
)
titanic_df.head()
```

We need to do some simple preprocessing before our neural network can deal with this data. 

```python
features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Cabin','Embarked']
titanic_df2 = titanic_df.loc[:,features]
titanic_df2['CabinLetter'] = titanic_df2['Cabin'].str.slice(0,1)
X = pd.get_dummies(titanic_df2.drop('Cabin',axis=1)).dropna()
# Standard scaling
means = X.mean()
sds = X.std()
X2 = X.apply(lambda x: (x-means)/sds,axis=1)
X2
```

```python
t = titanic_df.loc[X2.index,'Survived']
t
```

#### Problem 1
In your own words, describe the preprocessing steps I took above.


Your solution here.


### Training the network
We are now going to train the network. We'll use the defaults that I've set in the function, but feel free to change them around and see how you can do. I am going to show you how setting the seed makes a difference in training the algorithm.

```python
seeds = [0,1,2,3,4,5]
results = None
w = {}
X_test = {}
t_test = {}
for seed in seeds:
    w[seed],X_test[seed],t_test[seed],results1 = Lab3_helper.train(X2,t,seed=seed)
    if results is None:
        results = results1
    else:
        results = results.append(results1)
```

```python
import altair as alt
alt.data_transformers.disable_max_rows()

source = results.reset_index().drop(['n','test_size','val_size'],axis=1).melt(id_vars=['epoch','seed'])

alt.Chart(source).mark_line().encode(
    x='epoch',
    y=alt.Y('value',title='Accuracy'),
    color='variable',
    column='seed'
)
```

#### Problem 2
Run a similar experiment but vary the learning rate as below. Keep the seed constant (seed=0). What do the graphs tell you about the parameter ``n`` (i.e., $\eta$)?

```python
ns = [0.01,0.1]
results = None
# Your solution here
```

```python
source = results.reset_index().drop(['seed','test_size','val_size'],axis=1).melt(id_vars=['epoch','n'])

alt.Chart(source).mark_line().encode(
    x='epoch',
    y=alt.Y('value',title='Accuracy'),
    color='variable',
    column='n'
)
```

#### Exercise 1
The first step to evaluating any classification problem is establishing a baseline. Write a function that calculates the baseline accuracy if you predict the majority class on the test dataset.

```python
from sklearn.model_selection import train_test_split
import numpy as np

seeds = [0,1,2,3,4,5]
results = pd.DataFrame(columns=["seed","frac_max_class","accuracy_test","accuracy_train2","accuracy_val"]).set_index("seed")
for seed in seeds:
    np.random.seed(seed)
    X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.3)
    X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.3)

    ## Your solution in evaluate_baseline(...)
    frac_max_class,accuracy_test,accuracy_train2,accuracy_val = Lab3_helper.evaluate_baseline(t_test,t_train2,t_val)
    
    results = results.append(pd.Series([frac_max_class,accuracy_test,accuracy_train2,accuracy_val],index=results.columns,name=seed))

results
```

```python
import altair as alt

source = results.melt()

alt.Chart(source).mark_boxplot().encode(
    x='variable:N',
    y=alt.Y('value:Q',scale=alt.Scale(domain=(0.5, 0.7)))
)
```

#### Exercise 2
Write a function that makes predictions for an X matrix using the weights.

```python
w,X_test,t_test,results = Lab3_helper.train(X2,t,seed=0)

y_test = Lab3_helper.predict(w,X_test)
y_test
```

#### Exercise 3
Write a function that calculates the confusion matrix.

```python
y_test = Lab3_helper.predict(w,X_test)

cm = Lab3_helper.confusion_matrix(t_test,y_test,labels=[0,1])
cm
```

#### Exercise 4
Sensitivity, recall, hit rate, or true positive rate (TPR)

${\displaystyle \mathrm {TPR} ={\frac {\mathrm {TP} }{\mathrm {P} }}={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FN} }}=1-\mathrm {FNR} }$

specificity, selectivity or true negative rate (TNR)

${\displaystyle \mathrm {TNR} ={\frac {\mathrm {TN} }{\mathrm {N} }}={\frac {\mathrm {TN} }{\mathrm {TN} +\mathrm {FP} }}=1-\mathrm {FPR} }$

precision or positive predictive value (PPV)

${\displaystyle \mathrm {PPV} ={\frac {\mathrm {TP} }{\mathrm {TP} +\mathrm {FP} }}=1-\mathrm {FDR} }$

F1

${\displaystyle F_{1}={\frac {2}{\mathrm {recall^{-1}} +\mathrm {precision^{-1}} }}=2\cdot {\frac {\mathrm {precision} \cdot \mathrm {recall} }{\mathrm {precision} +\mathrm {recall} }}={\frac {\mathrm {tp} }{\mathrm {tp} +{\frac {1}{2}}(\mathrm {fp} +\mathrm {fn} )}}}$

Write a function that calculates accuracy, sensitivity/recall, specificity, precision, and F1

```python
stats = Lab3_helper.evaluation(cm,positive_class=1)
stats
```


```python
stats = Lab3_helper.evaluation(cm,positive_class=0)
stats
```

#### Exercise 5
Create a function that trains our neural network for each of the seeds and then returns variable importance of each feature as:

${\it importance}(w_i) = \frac{1}{|seeds|}\sum_{s \in seeds} \frac{\sqrt{w_i^2}}{max\left(\sqrt{w_0^2}, \sqrt{w_1^2} ... \sqrt{w_d^2}\right)}$

Basically, compute the variable importance for each seed and then average.

```python
seeds = [0,1,2,3,4,5]
importances = Lab3_helper.importance(X2,t,seeds)
importances.sort_values(ascending=False)
```

#### Problem 3: Compare these variable importances to the variable importances achieved by test-based permutation and train-based variable importances. 

To complete this problem, you will have to copy your previous lab's solutions to this notebook or Lab3_helper.py. From there, you can make the modifications necessary to train and evaluate the neural network instead of the Bayesian classifier.

```python
# Your solution here
```

**Your interpretation here**

```python
# Good job!
# Don't forget to push with ./submit.sh
```

#### Having trouble with the test cases and the autograder?

You can always load up the answers for the autograder. The autograder runs your code and compares your answer to the expected answer. I manually review your code, so there is no need to hide this from you.

```python
import joblib
answers = joblib.load(f"{home}/csc-466-student/tests/answers_Lab3.joblib")
answers.keys()
```

```python
answers['exercise_2']
```

```python

```
