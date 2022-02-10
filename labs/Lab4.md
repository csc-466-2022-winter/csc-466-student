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

# Lab 4

## Decision Tree

For this lab, we are going to implement a decision tree based on the C4.5 algorithm. C4.5 provides several improvements over ID3 though the base structure is very similar. C4.5 removed the restriction that features must be categorical by dynamically defining a discrete attribute (based on numerical variables) that partitions the continuous attribute value into a discrete set of intervals. C4.5 converts the trained trees (i.e. the output of the ID3 algorithm) into sets of if-then rules. 

We will start with our titanic dataset.

**Note:** Exercises can be autograded and count towards your lab and assignment score. Problems are graded for participation.

```python
from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab4_helper
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
X = titanic_df2.drop('Cabin',axis=1)#.dropna()
X['CabinLetter'] = X['CabinLetter'].fillna("?")
X['Pclass'] = X['Pclass'].astype(str)
X['SibSp'] = X['SibSp'].astype(str)
X['Parch'] = X['Parch'].astype(str)
X = X.dropna()
X
```

```python
X.dtypes
```

We will first implement ID3 before we move towards C4.5. This means we cannot handle numeric data such as ``Age`` and ``Fare``. We will bin these into 20 categories. I picked 20 after trying a few different values. At this point, I do not know if it is a good selection or bad. This is part of the reason we will switch to C4.5. 

```python
X2 = X.copy()
X2['Age'] = pd.cut(X2['Age'],bins=20).astype(str) # bin Age up
X2['Age'].value_counts()
```

```python
X2['Fare'] = pd.cut(X2['Fare'],bins=20).astype(str) # bin Age up
X2['Fare'].value_counts()
```

```python
t = titanic_df.loc[X2.index,'Survived']
t
```

#### Exercise 1
Construct a function called ``entropy`` that calculates the entropy of a set (Pandas Series Object)

```python
e1 = Lab4_helper.entropy(t)
e2 = Lab4_helper.entropy(X2['CabinLetter'])
e1,e2
```

#### Exercise 2
Write a function called ``gain`` that calculates the information gain after splitting with a specific variable (Equation 12.2 from Marsland).

```python
g1 = Lab4_helper.gain(t,X2['Sex'])
g2 = Lab4_helper.gain(t,X2['Pclass'])
g3 = Lab4_helper.gain(t,X2['Age'])
g1,g2,g3
```

#### Exercise 3
C4.5 actually uses the gain ratio which is defined as the information gain "normalized" (divided) by the entropy before the split. You have written everything you need here. Just put it together.

```python
gr1 = Lab4_helper.gain_ratio(t,X2['Sex'])
gr2 = Lab4_helper.gain_ratio(t,X2['Pclass'])
gr3 = Lab4_helper.gain_ratio(t,X2['Age'])
gr1,gr2,gr3
```

#### Exercise 4
Define a function called ``select_split`` that chooses the column to place in the decision tree. This function returns the column name and the gain ratio for this column.

```python
col,gain_ratio = Lab4_helper.select_split(X2,t)
col,gain_ratio
```

#### Exercise 5
Now put it all together and construct a function called ``make_tree`` that returns a tree in the format shown below. This function is a recursive function. Think carefully about how to debug recursion (i.e., grab yourself a debugger such as https://docs.python.org/3/library/pdb.html). Think carefully the base cases. 

```python
tree = Lab4_helper.make_tree(X2,t)
Lab4_helper.print_tree(tree)
```

<!-- #region -->
#### Exercise 6
Create a recrusive function called ``generate_rules`` that returns an array of the rules from a tree. A rule is the form of:
```python
[('Sex', 'male'),
 ('Age', '(20.315, 24.294]'),
 ('Embarked', 'S'),
 ('Pclass', 3),
 ('SibSp', 1),
 0]
```
A single rule has a type of list. The last element in the list is the prediction, which is Survived=0 in this example. The tuples that preceed the last element are the conditions. Put another way, the above rule is equivalent to:
```python
if Sex == 'male' and Age == '(20.315, 24.294]' and Embarked == 'S' and Pclass == 3 and SibSp == 1:
    predicted_value = 0
```
<!-- #endregion -->

```python
rules = Lab4_helper.generate_rules(tree)
rules[:5] # the first 5 rules
```

```python
rules
```

#### Exercise 7
Create an improved function to create a tree called ``make_tree2``. This function is a recursive function. This function must add support for numeric columns, and it must incorporate a parameter that battles overfitting called ``min_split_count``. Minimum split count is incorporated as an additional base case. To implement, check to see if you have at least min_split_count items (i.e., num_elements >= min_split_count to split). The biggest change comes with the addition of numeric columns (Age and Fare in their original format). Please refer to the Marsland textbook for details on handling numeric values. In short, you try all possible locations to divide a numeric variable. For example, if your column has the values:
```
values = [1,3,2,5]
sorted_values = [1,2,3,5]
possible_splits = [<1.5,<2.5,<4]
```
Please make sure you denote your splits like I am doing above and how they are printed below.

```python
tree2 = Lab4_helper.make_tree2(X,t,min_split_count=20)
Lab4_helper.print_tree(tree2)
```

#### Exercise 8
So how are we doing? We can put everything together and evaluate our solutions.

Create a function to make predictions called ``make_prediction``. Then use your Lab3_helper solutions to do some evaluations.

```python
default = 0
from sklearn.model_selection import train_test_split

X2_train, X2_test, t_train, t_test = train_test_split(X2, t, test_size=0.3, random_state = 0)
X_train, X_test = X.loc[X2_train.index], X.loc[X2_test.index]

tree_id3 = Lab4_helper.make_tree(X2_train,t_train)
rules_id3 = Lab4_helper.generate_rules(tree_id3)
tree_c45 = Lab4_helper.make_tree2(X_train,t_train)
rules_c45 = Lab4_helper.generate_rules(tree_c45)

y_id3 = X2_test.apply(lambda x: Lab4_helper.make_prediction(rules_id3,x,default),axis=1)
y_c45 = X_test.apply(lambda x: Lab4_helper.make_prediction(rules_c45,x,default),axis=1)
```

```python
import Lab3_helper
```

```python
# Evaluate the id3
cm_id3 = Lab3_helper.confusion_matrix(t_test,y_id3,labels=[0,1])
stats_id3 = Lab3_helper.evaluation(cm_id3,positive_class=1)
stats_id3
```

```python
# Evaluate the c45
cm_c45 = Lab3_helper.confusion_matrix(t_test,y_c45,labels=[0,1])
stats_c45 = Lab3_helper.evaluation(cm_c45,positive_class=1)
stats_c45
```

```python
source = pd.DataFrame.from_records([stats_id3,stats_c45])
source['Method'] = ['ID3','C4.5']
source
```

**Problem 1:** How do the two algorithms compare for this dataset?

Your answer here

**Problem 2:** Is this a robust experiment? How would you make it more robust? i.e., what are the flaws with what we did?

Your answer here

**Problem 3:** Repeat this experiment with min_split_count = 10, 20, 40, 80. How do the results change for C4.5?

Your answer here

```python
# Good job!
# Don't forget to push with ./submit.sh
```

<!-- #region -->
#### Having trouble with the test cases and the autograder?

You can always load up the answers for the autograder. The autograder runs your code and compares your answer to the expected answer. I manually review your code, so there is no need to hide this from you.

```python
import joblib
answers = joblib.load(f"{home}/csc-466-student/tests/answers_Lab4.joblib")
answers.keys()
```
<!-- #endregion -->

```python

```
