import copy
import json

import numpy as np
import pandas as pd

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

def entropy(y):
    e = None
    # YOUR SOLUTION HERE
    return e

def gain(y,x):
    g = 0
    # YOUR SOLUTION HERE
    return entropy(y) - g

def gain_ratio(y,x):
    # YOUR SOLUTION HERE
    return g/entropy(y)

def select_split(X,y):
    col = None
    gr = None
    # YOUR SOLUTION HERE
    return col,gr

def make_tree(X,y):
    tree = {}
    # Your solution here
    return tree

# if you want to print like me :)
def print_tree(tree):
    mytree = copy.deepcopy(tree)
    def fix_keys(tree):
        if type(tree) != dict:
            if type(tree) == np.int64:
                return int(tree)
        new_tree = {}
        for key in list(tree.keys()):
            if type(key) == np.int64:
                new_tree[int(key)] = tree[key]
            else:
                new_tree[key] = tree[key]
        for key in new_tree.keys():
            new_tree[key] = fix_keys(new_tree[key])
        return new_tree
    mytree = fix_keys(mytree)
    print(json.dumps(mytree, indent=4, sort_keys=True))

def generate_rules(tree):
    rules = []
    # Your solution here
    return rules

def select_split2(X,y):
    col = None
    gr = None
    return gr,col

def make_tree2(X,y,min_split_count=5):
    tree = {}
    # Your solution here
    return tree


def make_prediction(rules,x,default):
    # Your solution here
    return(default)
