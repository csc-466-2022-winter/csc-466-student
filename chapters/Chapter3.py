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
# # Chapter 3 - Single Layer Neural Networks
#
# Paul E. Anderson

# + slideshow={"slide_type": "skip"}
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
home = str(Path.home()) # all other paths are relative to this path. change to something else if this is not the case on your system

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Fruit Classification with Single Layer Perceptron
#
# Consider a pattern recognition problem where we want to sort fruit on an assembly line. We want all the apples to be sorted with all the apples, all the oranges with the oranges, etc. We have a variety of senor data at our displosal such as mass and width of the fruit. **How do we accomplish this and what knowledge can we extract?**
#
# One of the first goals of any KDD or data science problem is to perform an exploratory data analysis (EDA). Please refer to Chapter 2 for a more complete EDA.

# + slideshow={"slide_type": "fragment"}
import pandas as pd
import numpy as np

fruits = pd.read_csv(f'{home}/csc-466-student/data/fruit_data_with_colours.csv')
fruits.head() # Returns the first 5 rows of the data
# -

# ### Your brain is amazing
#
# > It deals with noisy and even inconsistent data, and produces answers that are usually correct from very high dimensional data (such as images) very quickly. All amazing for something that weighs about 1.5 kg and is losing parts of itself all the time (neurons die as you age at impressive/depressing rates), but its performance does not degrade appreciably (in the jargon, this means it is robust). - Marsland

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Strong AI vs Weak AI (narrow AI)
#
# Weak AI - focuses on a specific task (e.g., playing chess, picking out dogs and cats from photos). 
#
# Strong AI - focuses on a variety of functions and eventually teaches itself to solve for new problems
#
# We will focus entirely on KDD related to weak AI

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Inspiration often comes from biology
#
# At a high enough level of abstraction the basic building blocks of your brain are relatively easy to understand. These **neurons** fire when the input signal reaches a certain threshold. You have about 100 billion neurons. In more detail the process works as follows:
#
# >Neuron general operation is similar in all cases: transmitter chemicals within the fluid of the brain raise or lower the electrical potential inside the body of the neuron. If this membrane potential reaches some threshold, the neuron spikes or fires, and a pulse of fixed strength and duration is sent down the axon. The axons divide (arborise) into connections to many other neurons, connecting to each of these neurons in a synapse. Each neuron is typically connected to thousands of other neurons, so that it is estimated that there are about 100 trillion (= 1014) synapses within the brain. After firing, the neuron must wait for some time to recover its energy (the refractory period) before it can fire again. - Marsland

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Why do we call what we do learning?
#
# * One principal concept of learning is called **plasticity**
# * **Plasticity** is modifying the strength of synaptic connections between neurons (and creating new connections).
# * We don’t know all of the mechanisms by which the strength of these synapses gets adapted
# * We do know one that was first postulated by Donald Hebb in 1949

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Hebb's Rule
#
# * Rule states that changes in the strength of synaptic connections are proportional to the correlation in the firing of the two connecting neurons
#
# * Example 1: If two neurons consistently fire simultaneously, then connection between them becomes stronger.
#
# * Example 2: If the two neurons never fire simultaneously, the connection between them will die away. 

# + [markdown] slideshow={"slide_type": "subslide"}
# ### First mathematical model of a neuron: McCulloch and Pitts
#
# #### Neuron $j$
# <img src="https://www.tau.ac.il/~tsirel/dump/Static/knowino.org/w/images/thumb/ArtificialNeuronModel_english.png/350px-ArtificialNeuronModel_english.png">

# + [markdown] slideshow={"slide_type": "fragment"}
# > A picture of McCulloch and Pitts’ mathematical model of a neuron. The inputs xi are multiplied by the weights wi, and the neurons sum their values. If this sum is greater than the threshold θ then the neuron fires; otherwise it does not. - Marsland

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Limitations of McCulloch and Pitts
#
# We won't get into all of the limitations, but a few important ones to remember are:
#
# * Real neurons do not output a single output response, but a spike train (a sequence of pulses)
#
# * neurons don't actually respond as threshold devices, but produce a graded output in a continuous way
#
# * They do still have the transition between firing and not firing
#
# * Neurons are not updated according to a computer clock but update themselves asynchronously

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Designing a single layer neural network
#
# <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSXS6InFwhKDWTsBsW9WcDtyubH22eDXIcrWg&usqp=CAU">

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Simple things at first
#
# Let's try to create a single neuron that will predict between an apple and an orange.

# + slideshow={"slide_type": "subslide"}
fruits2 = fruits.loc[fruits['fruit_name'].isin(['orange','apple'])]
fruits2['fruit_name'].value_counts()

# + slideshow={"slide_type": "fragment"}
fruits2.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Create X and t

# + slideshow={"slide_type": "fragment"}
X = fruits2[['mass','width','height']]
X.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Stop and think: How do we make a variable called ``t`` that is 1 if it is an apple, 0 otherwise?

# + slideshow={"slide_type": "subslide"}
# Your solution here
t

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Let's try a single sample through a single neuron.
#
# We will begin by pulling out a single sample.

# + slideshow={"slide_type": "fragment"}
x = X.iloc[0]
x

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Does it make sense to scale the features?
#
# #### Stop and think: How would you scale each column in X so it has a mean of 0 and standard deviation of 1?

# + slideshow={"slide_type": "fragment"}
# Your solution here
X2.head()

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Does this look better?

# + slideshow={"slide_type": "fragment"}
x = X2.iloc[0]
x

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Now we need to choose our initial weights
#
# #### Stop and think: What should we set them to?

# + slideshow={"slide_type": "fragment"}
# Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# #### We need an activation function now

# + slideshow={"slide_type": "fragment"}
def activation(net,threshold=0):
    if net > threshold:
        return 1
    return 0


# + [markdown] slideshow={"slide_type": "subslide"}
# #### Now let's put it together!
#
# #### Stop and think: How would you use $w$, $x$, and the activation function to get a prediction?

# + slideshow={"slide_type": "fragment"}
# Your solution here

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Did we get this example correct?!!!

# + slideshow={"slide_type": "fragment"}
t.iloc[0] == activation(np.sum(w*x))

# + [markdown] slideshow={"slide_type": "fragment"}
# We did it! Case closed... OK. We should check out all of them.

# + slideshow={"slide_type": "subslide"}
y = X2.apply(lambda x: activation(np.sum(w*x)),axis=1)
y

# + slideshow={"slide_type": "subslide"}
sum(y == t)/len(y)

# + [markdown] slideshow={"slide_type": "fragment"}
# #### Well that is worse than guessing!!!!

# + [markdown] slideshow={"slide_type": "subslide"}
# ### We are finally to the big question
#
# **We are ready to learn! And we need rules!**
#
# <img src="https://i.pinimg.com/600x315/7d/d6/03/7dd603a62e8616251d26bdb856848c90.jpg">

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Here is our formula if y and t are integers (check the signs on a piece of paper)
#
# $w_{i} = w_{i} - \eta(y −t)·x_i$
#
# ```python
# n = 0.25
# w[0] = w[0] - n * (y - t) * x[0]
# w[1] = w[1] - n * (y - t) * x[1]
# w[2] = w[2] - n * (y - t) * x[2]
# ```
#
# But we have multiple y's and t's, so we need to loop through them! But first let's see what happens one time.

# + slideshow={"slide_type": "subslide"}
y[1],t[1]

# + slideshow={"slide_type": "fragment"}
n = 0.25
print(w[0] - n*(y[1] - t[1])*x[0])
print(w[1] - n*(y[1] - t[1])*x[1])
print(w[2] - n*(y[1] - t[1])*x[2])

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Did they go up or down? Why?

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Let's put it all together
#
# We are going to have choices!!!
#
# The first is what test set size and then what validation set size! We are going to try different values, so don't worry :)

# + slideshow={"slide_type": "subslide"}
from sklearn.model_selection import train_test_split
X_train, X_test, t_train, t_test = train_test_split(X2, t, test_size=0.3, random_state=0)
X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=0.5, random_state=0)

# + slideshow={"slide_type": "subslide"}
# Let's run it a bunch of times
n = 0.25 # We can try different values of the learning rate
nepochs = 10 # How many iterations should we run?
train_accuracy = []
val_accuracy = []
w = [1,1,1] # because why not?
for epoch in range(nepochs):
    y_train2 = None
    y_val = None
    # Your solution here
    
    train_accuracy.append(sum(t_train2 == y_train2)/len(t_train2))
    val_accuracy.append(sum(t_val == y_val)/len(t_val))
    
    for i in range(len(y_train2)):
        # Update those weights!
        # Your solution here
        
results = pd.DataFrame({"epoch": np.arange(nepochs)+1,"train_accuracy":train_accuracy,"val_accuracy":val_accuracy})

# + slideshow={"slide_type": "subslide"}
import altair as alt

source = results.melt(id_vars=['epoch'])

alt.Chart(source).mark_line().encode(
    x='epoch',
    y='value',
    color='variable'
)

# + slideshow={"slide_type": "subslide"}
y_test = X_test.apply(lambda x: activation(np.sum(w*x)),axis=1)
sum(y_test == t_test)/len(t_test)
# -


