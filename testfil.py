import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time as t

df = pd.read_csv("magic04.data")

X = df.to_numpy()[:,0:10]
Y = df.to_numpy()[:,10]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.90)


def getLabelSplitAvg(X : np.ndarray, y : list, col : int) -> Tuple[list, list]:
    xAvg = np.sum(X[:][col])/len(y)
    aboveEq = []
    below = []
    for i in range(len(y)):
        if X[i][col] >= xAvg:
            aboveEq.append(y[i])
        else:
            below.append(y[i])
    return (aboveEq, below)


def getLabelSplitAvgX(X : np.ndarray, y : list, col : int) -> Tuple[list, list]:
    """Only temporary, needs refactoring into getLabelSplitAvg"""
    xAvg = np.sum(X[:][col])/len(y)
    aboveEq = []
    below = []
    for i in range(len(y)):
        if X[i][col] >= xAvg:
            aboveEq.append(X[i])
        else:
            below.append(X[i])
    return (aboveEq, below)
# getLabelSplitAvg(X_train, Y_train, 0)


def entropy(y):
    if len(y) == 0:
        return 0
    probOfG = list(y).count('g')/(len(y)) 
    probOfH = 1 - probOfG
    if probOfG == 1 or probOfG == 0:
        return probOfG
    return (-1) * (probOfG * np.log2(probOfG) + (probOfH) * np.log2(probOfH))

def entropySplit(X : List[list], y : list):
    yEntropy = entropy(y)
    allEnt = []
    for col in range(len(X[0])):
        aboveEq, below = getLabelSplitAvg(X, y, col)
        allEnt.append(yEntropy - entropy(aboveEq)*len(aboveEq)/len(y) + entropy(below)*len(aboveEq)/len(y))
    return allEnt.index(max(allEnt))
    

def learn(X, y):
    if len(set(y))==1:
        return y[0]

    elif all(list(map(lambda x: all(list(map(lambda y: X[0][0]==y, x))), list(X)))):
        return 'g' if list(y).count('g')>(len(y)/2) else 'h'
    
    else:
        splitIndex = entropySplit(X,y)
        x1, x2 = getLabelSplitAvgX(X, y, splitIndex)
        y1, y2 = getLabelSplitAvg(X, y, splitIndex)

        return (learn(x1, y1), learn(x2, y2), splitIndex, np.sum(X[:, splitIndex])/len(y))

# tree = learn(X, y)
# predict(new_x, tree)

t = learn(X_test,Y_test)


def predict(x, tree):
    if tree in ['g', 'h']:
        return tree
    else:
        splitIndex = tree[2]
        avg = tree[3]
        return predict(x, tree[0]) if x[splitIndex] >= avg else predict(x, tree[1])

#g = learn(X_train, Y_train)
#print(predict([36.1741,17.6865,2.946,0.2865,0.1591,-4.7746,-18.9697,11.3256,0.254,191.455], g))

print(predict(X_train[0],t))
