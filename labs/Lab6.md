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

# Lab 6

## Choosing among parameters when clustering

### At the end of this lab, I should be able to
* Formulate your own clustering questions and understand how you can go about getting answers
* Understand how to select a clustering algorithm for your task

**Note:** Exercises can be autograded and count towards your lab and assignment score. Problems are graded for participation.

```python
from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system
```

```python
%load_ext autoreload
%autoreload 2

# make sure your run the cell above before running this
import Lab6_helper
```

```python
import numpy as np
```

## Our data
We will be using a well known housing dataset from Boston.
<pre>
The Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic
 prices and the demand for clean air', J. Environ. Economics & Management,
 vol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics
 ...', Wiley, 1980.   N.B. Various transformations are used in the table on
 pages 244-261 of the latter.

 Variables in order:
 CRIM     per capita crime rate by town
 ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
 INDUS    proportion of non-retail business acres per town
 CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
 NOX      nitric oxides concentration (parts per 10 million)
 RM       average number of rooms per dwelling
 AGE      proportion of owner-occupied units built prior to 1940
 DIS      weighted distances to five Boston employment centres
 RAD      index of accessibility to radial highways
 TAX      full-value property-tax rate per $10,000
 PTRATIO  pupil-teacher ratio by town
 B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
 LSTAT    % lower status of the population
 MEDV     Median value of owner-occupied homes in $1000's
</pre>

```python
import pandas as pd
df = pd.read_csv(f"{home}/csc-466-student/data/housing/boston_fixed.csv")
df.head()
```

**Problem 1.** Read the descriptions of the features above, and come up with 2-3 reasonable questions with corresponding methods to test them. The only one that you cannot write, is the one I write below:

Example questions: 
* Are there any definitive subgroupings (i.e., clusters) of towns in the dataset? 
* How many (if any) groups/clusters are there in the dataset?
* Are there any clusters of median value of owner-occupied homes? And if so, can we use the rest of the data to predict these clusters? 

Methodology:
1. Empirically determine the best clustering method from our known list of kmeans and hiearchical clustering.
2. Using this best clustering model, visualize the data using PCA
3. Apply clustering algorithms to MEDV and then use random forest to predict these clusters presenting the evaluation.


**YOUR SOLUTION HERE**


Overall question: Are there any clusters of towns? 

Use the following methodology:

1. Empirically determine the best clustering method from our known list kmeans and hiearchical clustering
2. Using this best clustering, visualize the data using PCA


**Exercise 1** A lot of methods depend on the scaling of data, so we need to decide on a scaling method. We will use the autoscaling method described in sklearn as:
"The standard score of a sample x is calculated as:

z = (x - u) / s

where u is the mean of the training samples or zero if with_mean=False, and s is the standard deviation of the training samples or one if with_std=False." - <a href="https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html">Source</a>

For this exercise, scale ``df`` using the StandardScaler in sklearn. For consistency with later code, call this new scaled dataframe ``X``.

```python
X = Lab6_helper.scale(df)
X
```

**Exercise 2** We now need to take a look at our data, but it is too many dimensions! For this task we need to reduce the dimension. Reduce the dataset down to two dimensions using PCA. <a href="https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html">Here is a link to the documentation.</a> Store the transformed data in a variable called ``X_pca``.

```python
X_pca = Lab6_helper.pca(X)
display(X_pca)
X_pca.plot.scatter(x=X_pca.columns[0],y=X_pca.columns[1]);
```

**Exercise 3** Our next major step is to apply kmeans to our data ``X`` (do NOT cluster on ``X_pca``) for several different values of ``k``. We'll compare these results later. The documentation for kmeans is <a href="https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html">here</a>. Fill in the loop that constructs the kmeans models for each of the values of ``k`` specified below.

```python
kmeans_models = Lab6_helper.kmeans(X,range_n_clusters = [2, 3, 4, 5, 6],random_state=10)
kmeans_models
```

**Exercise 4** Now we need assign cluster labels to each sample in our dataset. Fill in the following to accomplish this:

```python
cluster_labels = Lab6_helper.assign_labels(X,kmeans_models)
cluster_labels
```

We now have 5 different clusterings of our data. We need to know which one of these is the best. Let's visualize the clusters (k=2 and k=3) using the cluster_labels and PCA. <a href="https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.scatter.html">Here is some documentation on how to set the color.</a>

```python
colorings = {}
colorings[2] = cluster_labels[2].map({0: "Blue", 1: "Red"}) # This is a new pandas command for us that maps all 0 values to Blue, etc
colorings[3] = cluster_labels[3].map({0: "Blue", 1: "Red",2: "Pink"}) # This is a new pandas command for us that maps all 0 values to Blue, etc
X_pca.plot.scatter(x=X_pca.columns[0],y=X_pca.columns[1],c=colorings[2])
X_pca.plot.scatter(x=X_pca.columns[0],y=X_pca.columns[1],c=colorings[3])
colorings = {}
colorings[2] = cluster_labels[2].map({0: "Blue", 1: "Red"}) # This is a new pandas command for us that maps all 0 values to Blue, etc
colorings[3] = cluster_labels[3].map({0: "Blue", 1: "Red",2: "Pink"}) # This is a new pandas command for us that maps all 0 values to Blue, etc
```

### Choosing a $k$
We will now start assembling information we need to make a decision. There are many ways to evaluate clusters, but one of the best ways is through a silhouette score. Here is an excerpt of the documentation from sklearn:
"Silhouette analysis can be used to study the separation distance between the resulting clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring clusters and thus provides a way to assess parameters like number of clusters visually. This measure has a range of [-1, 1].

Silhouette coefficients (as these values are referred to as) near +1 indicate that the sample is far away from the neighboring clusters. A value of 0 indicates that the sample is on or very close to the decision boundary between two neighboring clusters and negative values indicate that those samples might have been assigned to the wrong cluster." - <a href="https://scikit-learn.org/dev/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#example-cluster-plot-kmeans-silhouette-analysis-py">Source</a>

```python
# The silhouette_score gives the average value for all the samples.
# This gives a perspective into the density and separation of the formed
# clusters
from sklearn.metrics import silhouette_score

n_clusters = 2
silhouette_avg = silhouette_score(X, cluster_labels[n_clusters])
print("For n_clusters =", n_clusters,
      "The average silhouette_score is :", silhouette_avg)
```

The following is pulled directly from https://en.wikipedia.org/wiki/Silhouette_(clustering).

For data point $i\in C_{i}$ (data point $i$ in the cluster $C_{i}$), let

${\displaystyle a(i)={\frac {1}{|C_{i}|-1}}\sum _{j\in C_{i},i\neq j}d(i,j)}$

be the mean distance between ${\displaystyle i}$ and all other data points in the same cluster, where ${\displaystyle d(i,j)}$ is the distance between data points ${\displaystyle i}$ and ${\displaystyle j}$ in the cluster ${\displaystyle C_{i}}$ (we divide by ${\displaystyle |C_{i}|-1}$ because we do not include the distance ${\displaystyle d(i,i)}$ in the sum). We can interpret ${\displaystyle a(i)}$ as a measure of how well ${\displaystyle i}$ is assigned to its cluster (the smaller the value, the better the assignment).

We then define the mean dissimilarity of point ${\displaystyle i}$ to some cluster ${\displaystyle C_{k}}$ as the mean of the distance from ${\displaystyle i}$ to all points in ${\displaystyle C_{k}}$ (where ${\displaystyle C_{k}\neq C_{i}}$).

For each data point ${\displaystyle i\in C_{i}}$, we now define

${\displaystyle b(i)=\min _{k\neq i}{\frac {1}{|C_{k}|}}\sum _{j\in C_{k}}d(i,j)}$

to be the smallest (hence the ${\displaystyle \min }$  operator in the formula) mean distance of ${\displaystyle i}$ to all points in any other cluster, of which ${\displaystyle i}$ is not a member. The cluster with this smallest mean dissimilarity is said to be the "neighboring cluster" of ${\displaystyle i}$ because it is the next best fit cluster for point ${\displaystyle i}$.

We now define a silhouette (value) of one data point ${\displaystyle i}$

${\displaystyle s(i)={\frac {b(i)-a(i)}{\max\{a(i),b(i)\}}}}$, if ${\displaystyle |C_{i}|>1}$
and ${\displaystyle s(i)=0}$, if ${\displaystyle |C_{i}|=1}$

Which can be also written as:

${\displaystyle s(i)={\begin{cases}1-a(i)/b(i),&{\mbox{if }}a(i)<b(i)\\0,&{\mbox{if }}a(i)=b(i)\\b(i)/a(i)-1,&{\mbox{if }}a(i)>b(i)\\\end{cases}}}$
From the above definition it is clear that

${\displaystyle -1\leq s(i)\leq 1}$



**Exercise 5** Write your own silhouette_scores function that returns $s(i)$ for each sample.

```python
scores = Lab6_helper.silhouette_scores(X,cluster_labels[n_clusters])
scores[:10]
```

```python
np.mean(scores) # do you match the sklearn implementation?
```

### Creating our plots
Let's put it all together and grab the scores for each cluster. I'll take over the plotting here. 

```python
s_df = pd.DataFrame(index=X.index,columns=cluster_labels.columns)
for k in s_df.columns:
    s_df.loc[:,k] = Lab6_helper.silhouette_scores(X,cluster_labels[k])
s_df
```

```python
s_df.index.name="i"
s_df = s_df.reset_index()
s_df
```

```python
source = s_df.melt(id_vars=["i"])
source.columns = ["i","k","s"]

import altair as alt
alt.Chart(source).mark_bar().encode(
    x = "s:Q",
    y = alt.Y("i:N",sort='x',axis=alt.Axis(labels=False)),
    row = "k:N",
    color = "k:N"
).resolve_scale(y='independent').properties(height=200)
```

#### Problem 2: What are the average silhouttee scores for each value of $k$? Can you relate this average value to what you are seeing in the above plot? What kind of shape are we looking for?

#### Your solution here


## Hiearchical Clustering

From here on out there are several problems and only one exercise.


**Problem 3:** That was kmeans clustering. What about hiearchical clustering? For this excercise, use the same ``X`` data and create a dendrogram using hiearchical clustering. <a href="https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html">Here is a link to a sample</a>. After you dig into this code, answer what kind of linkage method was used (answer with more than just the name)?

```python
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

# setting distance_threshold=0 ensures we compute the full tree.
model = AgglomerativeClustering(distance_threshold=0,n_clusters=None)

model = model.fit(X)
plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(model, truncate_mode='level', p=2)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")
plt.show()

# YOUR SOLUTION HERE
```

**Problem 4** Now change the linkage method to single linkage, and compare the plots. Are they better or worse?

```python
# YOUR SOLUTION HERE
```

### Clustering a single column to produce buckets


Now we are going to switch gears and cluster the ``MEDV`` column. First, we will create a density plot of ``MEDV``. Make sure you go back to the original dataframe ``df`` at this point.

```python
ax = df["MEDV"].plot.density();
ax.set_xlabel('MEDV');
```

**Exercise 6** To me it looks reasonable that there might be 3 clusters as we have the shoulder sticking out around 30 and the bump at around 50. Using kmeans and k=3, group each town in one of three clusters using the algorithm. 

```python
clusterer = Lab6_helper.bin_x(df[["MEDV"]])
labels = clusterer.predict(df[["MEDV"]])
df["y"] = labels
display(df)
df.groupby("y").MEDV.mean()
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
answers = joblib.load(f"{home}/csc-466-student/tests/answers_Lab6.joblib")
answers.keys()
```
<!-- #endregion -->

```python

```
