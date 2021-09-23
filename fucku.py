"""import dataclasses as dataclass
import node as Node


@dataclass
class Tree():
    root = Node(data="")

    def get_avg(col, x):
        c = x[:, col]
        avg = c.mean(axis=0)
        return avg

    def split_data(data, split_percent):
        split_value = len(data) * split_percent * 0.01
        test = data.sample(int(split_value))
        train = data.drop(test.index)

        return test, train

        X = magic_data.values[:, 0:11]
        Y = magic_data.values[:, 0]

        print("x: ", X)
        print("y: ", Y)
        print(test)
        print(train)
        print("---------------debug-----------------")
        print("split percentage:", split_percent, "\nsplit value:", split_value, "\nlength of dataset", len(
            magic_data), "\nlength of training data:", len(train), "\nlength of test data:", len(test))

    def is_identical(x):
        for row in x:
            if np.any(row != x[len(x)-1]):
                return False
            else:
                return True
"""
