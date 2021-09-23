#!/usr/bin/env python
# coding: utf-8

# In[991]:


import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time as t
import sys


# In[992]:


def split_data(data, split_percent):
    test_data = data.sample(frac=split_percent)
    training_data = data.drop(test_data.index)
    x_test = test_data.to_numpy()[:, 0:10]
    x_train = training_data.to_numpy()[:, 0:10]
    y_test = test_data.to_numpy()[:, 10]
    y_train = training_data.to_numpy()[:, 10]

    return x_train, x_test, y_train, y_test


# In[993]:

def is_identical(x):
    for row in x:
        if len(x) <= 1:
            return True
        elif row != x[len(x)-1]:
            return False
        else:
            return True


# In[994]:


df = pd.read_csv("magic04.data")

X = df.to_numpy()[:, 0:10]
Y = df.to_numpy()[:, 10]


## X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.75)
X_train, X_test, Y_train, Y_test = split_data(df, 0.75)
print("\nX_train:", X_train, "\nX_test:", X_test,
      "\nY_train:", Y_train, "\nY_test:", Y_test)


# In[995]:


test = X_train[:, 1]
test2 = X_train
# print(sorted(test))
print(type(X_train[0, 0]))


# En kolonne med en verdi for hver rad, hvor mange kilo bifangst av den arten for den turen per rekevekt for den samme turen. -> oversikt per kvartal, slå samme alle de av samme art. Men da summerer den opp de vektene, og det blir feil da vil jeg få et feil bilde?!?!

# In[996]:


def getLabelSplitAvg(X: np.ndarray, y: list, col: int) -> Tuple[list, list]:
    xAvg = np.sum(X[:][col])/len(y)
    aboveEq = []
    below = []
    for i in range(len(y)):
        if X[i][col] >= xAvg:
            aboveEq.append(y[i])
        else:
            below.append(y[i])
    return (aboveEq, below)


def getLabelSplitAvgX(X: np.ndarray, y: list, col: int) -> Tuple[list, list]:
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


# In[997]:


def entropy(y):
    if len(y) == 0:
        return 0
    probOfG = list(y).count('g')/(len(y))
    probOfH = 1 - probOfG
    if probOfG == 1 or probOfG == 0:
        return probOfG
    return (-1) * (probOfG * np.log2(probOfG) + (probOfH) * np.log2(probOfH))


def entropySplit(X: List[list], y: list):
    yEntropy = entropy(y)
    allEnt = []
    for col in range(len(X[0])):
        aboveEq, below = getLabelSplitAvg(X, y, col)
        allEnt.append(yEntropy - entropy(aboveEq)*len(aboveEq) /
                      len(y) + entropy(below)*len(aboveEq)/len(y))
    return allEnt.index(max(allEnt))


# In[998]:


def gini_index(y):
    probOfG = list(y).count('g')/(len(y))
    if not y:
        return 1
    return -probOfG * (1-(probOfG)) - (1-probOfG) * (1-(1-probOfG))


# In[999]:


print("Entropy:", entropy(['g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g',
      'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'h', 'h', 'h', 'h', 'h', 'h']))
print("Gini index:", gini_index(['g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h',
      'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'g', 'h', 'h', 'h', 'h', 'h', 'h', 'h']))
print("Gini index test:", gini_index(
    ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g']))
print("Gini index test 2:", gini_index(['g', 'h']))
print("Entropy test:", entropy(
    ['g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', ]))
print("Entropy test 2:", entropy(['g', 'h', ]))

print("Entropy split:", entropySplit(X_train, Y_train))


# In[1000]:


def learn(X, y, impurity_measure='entropy'):
    if is_identical(y):
        # sys.stdout.write(str(len(y)))
        # sys.stdout.flush()
        return y[0]

    elif all(list(map(lambda x: all(list(map(lambda y: X[0][0] == y, x))), list(X)))):
        return 'g' if list(y).count('g') > (len(y)-1/2) else 'h'

    else:
        splitIndex = entropySplit(X, y)
        x1, x2 = getLabelSplitAvgX(X, y, splitIndex)
        y1, y2 = getLabelSplitAvg(X, y, splitIndex)
        sys.stdout.write("[")
        sys.stdout.write(str(splitIndex))
        sys.stdout.write("]")
        sys.stdout.flush()

        return (learn(x1, y1), learn(x2, y2), splitIndex, np.sum(X[:, splitIndex])/len(y)-1)

# tree = learn(X, y)
# predict(new_x, tree)


t = learn(X_test, Y_test)


# In[ ]:


def predict(x, tree):
    if tree in ['g', 'h']:
        return tree
    else:
        splitIndex = tree[2]
        avg = tree[3]
        return predict(x, tree[0]) if x[splitIndex] >= avg else predict(x, tree[1])

#g = learn(X_train, Y_train)
#print(predict([36.1741,17.6865,2.946,0.2865,0.1591,-4.7746,-18.9697,11.3256,0.254,191.455], g))


print(predict(X_train[0], t))
