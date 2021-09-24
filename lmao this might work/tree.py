from dataclasses import dataclass
from node import Node
import numpy as np


@dataclass
class Tree():
    root: Node = Node()
    leaf_array = []

    def get_average(self, col, X):
        c = X[:, col]
        average = c.mean()
        return average

    def print(self):
        self.root.print(0)

    def get_most_common_y(self, y):
        count = np.unique(y)
        return count[0]

    def is_x_identical(self, X):
        for row in X:
            if np.any(row != X[0]):
                return False
        return True

    def is_label_identical(self, y):
        if np.all(y) or not np.any(y):
            self.root.y = y[0]
            return

    def get_column(self, col, X):
        return X[:, col]

    def less(self, x, y):
        return x <= y

    def greater(self, x, y):
        return x > y

    def split(self, X, y, function, col, split):
        rows = []
        for i, row in enumerate(X):
            if function(row[col], split):
                rows.append(np.append(row, y[i]))
        return np.array(rows)

    def probability(self, X, function, split, col, y):
        """"
        y: input
        Y: selected Y
        _y: y zero
        """
        column = self.get_column(col, X)
        rows = [i for i, val in enumerate(
            column) if function(val, split)]
        Y = [val for i, val in enumerate(y) if i in rows]
        _y = [y for y in Y if y == 0]
        if len(Y) == 0:
            return 0.5
        return len(_y)/len(Y)

    def calculate_information_gain(self, split, X, y, function, col, impurity_measure):
        probability = self.probability(X, function, split, col, y)
        if probability == 0 or probability == 1:
            return 0
        if impurity_measure == "entropy":
            return self.entropy(probability)
        elif impurity_measure == "gini":
            return self.gini_index(probability)

    def information_gain(self, col, split, X, impurity_measure, y):
        information_gain = self.calculate_information_gain(split, X, y, self.less, col, impurity_measure) -\
            (self.calculate_information_gain(
                split, X, y, self.greater, col, impurity_measure))
        return information_gain

    def best_information_gain(self, X, impurity_measure, y):
        _rows, cols = X.shape
        col_gains = []
        for col in range(cols):
            split_value = self.get_average(col, X)
            gain = self.information_gain(
                col, split_value, X, impurity_measure, y)
            col_gains.append((col, gain, split_value))
        return max(col_gains, key=lambda tup: tup[1])

    def entropy(self, probability):
        p2 = 1 - probability
        if probability == 1 or p2 == 0:
            return probability
        return (-1) * (probability * np.log2(probability) + (p2) * np.log2(p2))

    def gini_index(self, probability):
        '''TODO: Test'''
        return probability * (1-(probability)) + (1-probability) * (1-(1-probability))

    def accuracy(self, X_prune, y_prune):
        '''TODO: THIS NEEDS TO BE REFACTORED FOR SUBMISSION!'''
        '''return accuracy of the current tree, given test data X and y'''
        wrong = 1
        correct = 0
        for i, x in enumerate(X_prune):
            predicted_val = self.predict(self.root, x)
            if predicted_val == y_prune[i]:
                correct += 1
            else:
                wrong += 1
        accuracy = correct/wrong
        return accuracy

    def learn(self, X, y, node, X_prune, y_prune, prune=False, impurity_measure="entropy"):
        if node is None:
            raise Exception("Node is None")
        self.is_label_identical(y)
        self.is_x_identical(X)

        best_col, _, split_value = self.best_information_gain(
            X, impurity_measure, y)
        node.majority_label = self.get_most_common_y(y)
        node.data = split_value
        node.column = best_col
        node.left = Node()
        node.right = Node()

        lXy = self.split(X, y, self.less, best_col, split_value)
        rXy = self.split(X, y, self.greater, best_col, split_value)
        if lXy.size == 0 or rXy.size == 0:
            node.left = None
            node.right = None
            node.y = self.get_most_common_y(y)
            return
        # TODO: Make this use the data dataclass
        lX = lXy[:, :10]
        ly = lXy[:, 10]
        rX = rXy[:, :10]
        ry = rXy[:, 10]
        self.learn(lX, ly, node.left, X_prune, y_prune,
                   prune=prune, impurity_measure=impurity_measure)
        self.learn(rX, ry, node.right, X_prune, y_prune,
                   prune=prune, impurity_measure=impurity_measure)
        if prune:
            self.prune(X, y, X_prune, y_prune, node)

    def prune(self, X, y, X_prune, y_prune, node: Node):
        if node.right is None or node.left is None:
            return

        accuracy = self.accuracy(X_prune, y_prune)
        old_left = node.left
        old_right = node.right
        node.left = None
        node.right = None
        node.y = node.majority_label

        leaf_accuracy = self.accuracy(X_prune, y_prune)

        if leaf_accuracy >= accuracy:
            self.leaf_array.append([leaf_accuracy, accuracy])
        else:
            node.left = old_left
            node.right = old_right
            node.y = None

    def predict(self, node, x):
        split_val = node.data
        col = node.column
        if node.left is None or node.right is None:
            return node.y
        if x[col] < split_val:
            return self.predict(node.left, x)
        else:
            return self.predict(node.right, x)
