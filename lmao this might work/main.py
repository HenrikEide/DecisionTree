from dataclasses import dataclass
import pandas as pd
import numpy as np
from freshNode import Node
from tree import Tree


@dataclass
class Data():
    test_data: list
    training_data: list
    test_X: list
    test_Y: list
    train_X: list
    train_Y: list


def split_data(magic_data, split_percent):
    split_value = int(len(magic_data) * (1-split_percent))
    print(split_value)
    test = magic_data.sample(split_value)
    train = magic_data.drop(test.index)

    te_X = test.values[:, 0:10]
    tr_X = train.values[:, 0:10]
    te_Y = test.values[:, 10]
    tr_Y = train.values[:, 10]
    data = Data(test_data=test, training_data=train, test_X=te_X,
                test_Y=te_Y, train_X=tr_X, train_Y=tr_Y)

    return data
    print("x: ", X)
    print("y: ", Y)
    print(test)
    print(train)
    print("---------------debug-----------------")
    print("split percentage:", split_percent, "\nsplit value:", split_value, "\nlength of dataset", len(
        magic_data), "\nlength of training data:", len(train), "\nlength of test data:", len(test))


def get_data(dataset="lmao this might work\magic04.data", separator=',', header_size=None):
    data = pd.read_csv(dataset, sep=separator, header=header_size)

    ## print("Data length:", len(data))
    ## print("Data shape:", data.shape)
    ## print("Dataset:", data.head())

    data = split_data(data, 0.25)
    return data


# def entropy(y):
#     if len(y) == 0:
#         return 0
#     probOfG = list(y).count('g')/(len(y))
#     probOfH = 1 - probOfG
#     if probOfG == 1 or probOfG == 0:
#         return probOfG
#     return (-1) * (probOfG * np.log2(probOfG) + (probOfH) * np.log2(probOfH))


# def gini_index(y):
#     probOfG = list(y).count('g')/(len(y))
#     if not y:
#         return 1
#     return -probOfG * (1-(probOfG)) - (1-probOfG) * (1-(1-probOfG))


# def probability(X, function, split, col, y, tree):
#     """"
#     y: input
#     Y: selected Y
#     _y: y zero
#     """
#     column = tree.get_column(col, X)
#     rows = [i for i, val in enumerate(column) if function(val, split)]
#     Y = [val for i, val in enumerate(y) if i in rows]
#     _y = [y for y in Y if y == 0]
#     if len(Y) == 0:
#         return 0.5
#     return len(_y)/len(Y)


# def calculate_information_gain(split, X, y, function, col, impurity_measure):
#     probability = probability(X, function, split, col, y)
#     if probability == 0 or probability == 1:
#         return 0
#     if impurity_measure == "entropy":
#         return entropy(y)
#     elif impurity_measure == "gini":
#         return gini_index(y)


# def information_gain(col, split, X, impurity_measure, y, tree):
#     information_gain = calculate_information_gain(split, X, tree.less, col, impurity_measure, y) -\
#         (calculate_information_gain(split, X, tree.greater, col, impurity_measure, y))
#     return information_gain


# def best_information_gain(X, tree, impurity_measure, y):
#     _rows, cols = X.shape
#     col_gains = []
#     for col in cols:
#         split_value = tree.get_avg(col, X)
#         gain = information_gain(col, split_value, X, impurity_measure, y, tree)
#         col_gains.append(col, gain, split_value)
#     return max(col_gains, key=lambda tup: tup[1])
if __name__ == "__main__":

    data = get_data()
    # TODO: add time
    print(f"Len training data: {len(data.training_data)}\nlen test data: {len(data.test_data)}")
    tree = Tree()
    tree.learn(data.train_X, data.train_Y, tree.root, data.test_X,
               data.test_Y, prune=True, impurity_measure="entropy")
    print("Learn finished.")
    wrong = 0
    correct = 0
    for i, x in enumerate(data.train_X):
        prediction = tree.predict(tree.root, x)
        # print(prediction)
        if prediction == data.train_Y[i]:
            correct += 1
        else:
            wrong += 1
    

    # TODO: The whole main is copy paste, needs refactor
    print("training data length:", len(data.training_data),
          "testing data length:", len(data.test_data))
    print(f"\nPrediction.\nWrong: {wrong}\nCorrect: {correct}")
    print(f"accuracy {correct/(correct+wrong)}")

    '''testing on test data'''
    print("------\ntesting data:")
    wrong = 0
    correct = 0
    for i, x in enumerate(data.test_X):
        predicted_val = tree.predict(tree.root, x)
        if predicted_val == data.test_Y[i]:
            correct += 1
        else:
            wrong += 1
    print(f"correct predictions: {correct}, wrong predictions: {wrong}")
    print(f"accuracy {correct/(correct+wrong)}")
    # test_node = Node(data="hello_world")

    # print("t_data:", data.test_data, "tr_data:", data.training_data)
    # print("x_stuff:", data.test_X, "x_labels:", data.test_Y)

    # super_omega_tree = Tree(test_node)

    # print(test_node.data, super_omega_tree)
