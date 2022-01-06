import numpy as np

def compute_priors(y):
    priors = {}
    return priors

def specific_class_conditional(x,xv,y,yv):
    prob = None
    return prob

def class_conditional(X,y):
    probs = {}
    return probs

def posteriors(probs,priors,x):
    post_probs = {}
    return post_probs

def train_test_split(X,y,test_frac=0.5):
    inxs = list(range(len(y)))
    np.random.shuffle(inxs)
    X = X.iloc[inxs,:]
    y = y.iloc[inxs]
    Xtrain,ytrain,Xtest,ytest = None,None,None,None
    return Xtrain,ytrain,Xtest,ytest

def exercise_6(Xtrain,ytrain,Xtest,ytest):
    accuracy = None
    return accuracy

def exercise_7(Xtrain,ytrain,Xtest,ytest, npermutations = 10):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtest2 = Xtest.copy()
            Xtest2[col] = Xtest[col].sample(frac=1, replace=False).values
        importances[col] = importances[col]/npermutations
    return importances

def exercise_8(Xtrain,ytrain,Xtest,ytest, npermutations = 20):
    # initialize what we are going to return
    importances = {}
    for col in Xtrain.columns:
        importances[col] = 0
    # find the original accuracy
    orig_accuracy = exercise_6(Xtrain,ytrain,Xtest,ytest)
    # now carray out the feature importance work
    for col in Xtrain.columns:
        for perm in range(npermutations):
            Xtrain2 = Xtrain.copy()
            Xtrain2[col] = Xtrain[col].sample(frac=1, replace=False).values
        importances[col] = importances[col]/npermutations
    return importances
