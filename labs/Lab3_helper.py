from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

def activation(net):
    return 1/(1+np.exp(-net))

def train(X,t,nepochs=200,n=0.5,test_size=0.3,val_size=0.3,seed=0):
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=test_size,random_state=seed)
    X_train2, X_val, t_train2, t_val = train_test_split(X_train, t_train, test_size=val_size,random_state=seed)

    train_accuracy = []
    val_accuracy = []
    nfeatures = X.shape[1]
    np.random.seed(seed)
    w = 2*np.random.uniform(size=(nfeatures,)) - 1
    
    for epoch in range(nepochs):
        y_train2 = X_train2.apply(lambda x: activation(np.dot(w,x)),axis=1)
        y_val = X_val.apply(lambda x: activation(np.dot(w,x)),axis=1)

        train_accuracy.append(sum(t_train2 == np.round(y_train2))/len(t_train2))
        val_accuracy.append(sum(t_val == np.round(y_val))/len(t_val))
                
        for j in range(len(w)):
            w[j] -= n*np.dot((y_train2 - t_train2)*y_train2*(1-y_train2),X_train2.iloc[:,j])
            
    results = pd.DataFrame({"epoch": np.arange(nepochs)+1, 'train_accuracy':train_accuracy,'val_accuracy':val_accuracy,
                            "n":n,'test_size':test_size,'val_size':val_size,'seed':seed
                           }).set_index(['n','test_size','val_size','seed'])
    return w,X_test,t_test,results

def evaluate_baseline(t_test,t_train2,t_val):
    frac_max_class = None
    accuracy_test = None
    accuracy_train2 = None
    accuracy_val = None
    return frac_max_class,accuracy_test,accuracy_train2,accuracy_val

def predict(w,X,threshold=0.5):
    y = None
    return y

def confusion_matrix(t,y,labels):
    cm = pd.DataFrame(columns=labels,index=labels)
    # actual is on the rows, pred on the columns
    return cm

def evaluation(cm,positive_class=1):
    stats = {}
    return stats

def importance(X,t,seeds):
    importances = pd.Series(np.zeros((X.shape[1],)),index=X.columns)
    return importances
