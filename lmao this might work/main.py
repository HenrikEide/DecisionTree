from dataclasses import dataclass
import pandas as pd
from tree import Tree
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


@dataclass
class Data():
    test_data: list
    training_data: list
    test_X: list
    test_Y: list
    train_X: list
    train_Y: list
    pruning_X: list
    pruning_Y: list


def split_data(magic_data, split_percent):
    split_value = int(len(magic_data) * (1-split_percent))
    test = magic_data.sample(split_value)
    train = magic_data.drop(test.index)
    split_value = int(len(train) * (1-split_percent))
    pruning = train.sample(split_value)
    train = train.drop(pruning.index)

    te_X = test.values[:, 0:10]
    tr_X = train.values[:, 0:10]
    pr_X = pruning.values[:, 0:10]
    te_Y = test.values[:, 10]
    tr_Y = train.values[:, 10]
    pr_Y = pruning.values[:, 10]
    data = Data(test_data=test, training_data=train, test_X=te_X,
                test_Y=te_Y, train_X=tr_X, train_Y=tr_Y, pruning_X=pr_X, pruning_Y=pr_Y)

    return data


def get_data(dataset="lmao this might work\magic04.data", separator=',', header_size=None, split=0.25):
    data = pd.read_csv(dataset, sep=separator, header=header_size)
    data = split_data(data, split)
    return data


def get_best_tree(tree_list, data):
    tree_names = ["Tree with entropy without pruning", "Tree with entropy and pruning",
                  "Tree with gini index without pruning", "Tree with gini index with pruning"]
    best_accuracy = 0
    best_tree = 0
    tree_scores = []
    for i, tree in enumerate(tree_list):
        accuracy = 0
        wrong = 0
        correct = 0
        for j, x in enumerate(data.train_X):
            prediction = tree.predict(tree.root, x)
            if prediction == data.train_Y[j]:
                correct += 1
            else:
                wrong += 1
        accuracy = correct/(correct+wrong)
        tree_scores.append((tree_names[i], accuracy))
        if best_accuracy < accuracy:
            print(f"new best accuracy found: {accuracy}")
            best_accuracy = accuracy
            best_tree = i
    return tree_list[best_tree], tree_names[i], tree_scores


if __name__ == "__main__":
    data = get_data(split=0.25)
    tree1 = Tree()
    tree2 = Tree()
    tree3 = Tree()
    tree4 = Tree()

    # sklearn's decision tree for comparison
    timestart = time.time()
    sktree = DecisionTreeClassifier()
    sktree = sktree.fit(data.train_X, data.train_Y)
    skpreds = sktree.predict(data.test_X) 
    sktime = time.time() - timestart

    print("Starting entropy tree learning...")
    tree1.learn(data.train_X, data.train_Y, tree1.root, data.pruning_X,
                data.pruning_Y, prune=False, impurity_measure="entropy")
    print("Entropy tree learning done...")
    print("Starting entropy and pruning tree...")
    tree2.learn(data.train_X, data.train_Y, tree2.root, data.pruning_X,
                data.pruning_Y, prune=True, impurity_measure="entropy")
    print("Entropy and pruning tree learning done...")
    print("Starting gini index tree learning...")
    tree3.learn(data.train_X, data.train_Y, tree3.root, data.pruning_X,
                data.pruning_Y, prune=False, impurity_measure="gini")
    print("Gini index tree learning done...")
    print("Starting gini index with pruning tree learning...")
    tree4.learn(data.train_X, data.train_Y, tree4.root, data.pruning_X,
                data.pruning_Y, prune=True, impurity_measure="gini")
    print("Gini index and pruning tree learning done...")
    print("...getting most accurate tree...")

    tree_to_test, best_tree, tree_scores = get_best_tree(
        [tree1, tree2, tree3, tree4], data)

    print("------testing data------")
    print(best_tree, "was the best tree")
    start_time = time.time()
    wrong = 0
    correct = 0
    for i, x in enumerate(data.test_X):
        predicted_val = tree_to_test.predict(tree_to_test.root, x)
        if predicted_val == data.test_Y[i]:
            correct += 1
        else:
            wrong += 1
    print(
        f"correct/wrong predictions: {correct}/{wrong}\naccuracy {correct/(correct+wrong)} in {time.time()-start_time} seconds")
    
    print(f"\nSklearn's decision tree for comparison:\nAccuracy: {accuracy_score(data.test_Y, skpreds)} in {sktime} seconds")
    
