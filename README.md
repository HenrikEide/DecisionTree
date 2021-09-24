## Decision Tree implementation

### How to use:

`learn(X, y, prune, impurity_measure)`
Train algorithm on attribute values `X` with class labels `y`. Set prune to True to enable pruning, and set 
impurity_measure to either "entropy" or "gini". Defaults: pruning off and entropy for impurity measure.

`predict(x, tree)`
Predict label given new data point `x` and decision tree `tree`

main.py
To use our already set settings with the magic04 data set, run the main.py file. To change train/test split ratio change the split parameter sent to get_data() in the first line of the "__main__" block in the file.
