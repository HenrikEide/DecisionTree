from dataclasses import dataclass
import pandas as pd
import numpy as np
from freshnode import Node
from tree import Tree
import time


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
    test = magic_data.sample(split_value)
    train = magic_data.drop(test.index)

    te_X = test.values[:, 0:10]
    tr_X = train.values[:, 0:10]
    te_Y = test.values[:, 10]
    tr_Y = train.values[:, 10]
    data = Data(test_data=test, training_data=train, test_X=te_X,
                test_Y=te_Y, train_X=tr_X, train_Y=tr_Y)

    return data


def get_data(dataset="magic04.data", separator=',', header_size=None, split=0.25):
    data = pd.read_csv(dataset, sep=separator, header=header_size)
    data = split_data(data, split)
    return data


if __name__ == "__main__":

    data = get_data(split=0.5)
    tree = Tree()
    start_time = time.time()
    tree.learn(data.train_X, data.train_Y, tree.root, data.test_X,
               data.test_Y, prune=True, impurity_measure="gini")
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
    print(
        f"training data length: {len(data.training_data)}\ntesting data length:, {len(data.test_data)}")
    print(
        f"correct/wrong predictions: {correct}/{wrong}\naccuracy {correct/(correct+wrong)} in {time.time()-start_time} seconds")
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
    print(
        f"correct/wrong predictions: {correct}/{wrong}\naccuracy {correct/(correct+wrong)} in {time.time()-start_time} seconds")
