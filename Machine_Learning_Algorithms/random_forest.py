from collections import Counter

import numpy as np

from decision_tree import DecisionTree


# The random forest algorith can be viewed as an extension of the decision tree algorithm/model
# Random forest algorithm is one of tge most powerful and most popular algorithms
# Idea is to combine multiple trees into a forest
# Each tree gets a random subset of the training data --> hence the word random
# We then make a prediction with each of the trees at the end
# Finally we make a majority vote then to get the final prediction
# Random forest advantages
#   By building more trees, we have more chances to get the correct prediction
#   We also reduce the chance of overfitting with a single tree
#   Typically the accuracy of a random forest > than with a single tree

# Create a global function that will provide us with a random subset of training data --> bootstrapping
# X = training samples
# Y = training labels

def bootstrap_sample(X, y):
    # Get the amount/number of samples we have
    n_samples = X.shape[0]
    # The following will make a random choice between 0 and the number of samples
    # Replace = true means some indices can appear multiple times whereas others get dropped
    indices = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[indices], y[indices]


# Create another global helper function that will return to us the most common label
# Function will take in a vector of the class labels
def _most_common_label(y):
    # Create a counter obj
    # Will calculate all the number of occurences for all the y's
    counter = Counter(y)
    # Only interested in the first most common label
    # This returns a tuple containing the value as well as the number of occurences
    # We are only concerned with the value so we extract the 0th element
    _most_common = counter.most_common(1)[0][0]
    return _most_common


class RandomForest:

    # Make sure to store all our arguments into the designated fields of the object created

    def __init__(self, num_trees=100, min_samples_split=2, max_depth=100, n_feats=None):
        self.num_trees = num_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats

        # Lastly instantiate an empty array of trees where we want to store each single tree we now are going to create
        self.trees = []

    # This method will fit in all our training data and labels

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # Y = 1D row vector of size M (training labels)

    # X = training samples
    # Y = training labels

    def fit(self, X, y):
        # Make sure we start off with an empty array of trees
        self.trees = []

        # Now we can start training our trees
        # Make sure to build the designated number of trees
        for _ in range(self.num_trees):
            # Make sure to pass in all the features we have created previously
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth,
                                n_feats=self.n_feats)

            # After creating our tree, we now want to give it a random subset of training data
            X_sample, y_sample = bootstrap_sample(X, y)
            # Fit the tree with the random subset of training data provided
            tree.fit(X_sample, y_sample)
            # Now we want to append/add the newly created tree to our trees array
            self.trees.append(tree)

        # this is it for our training phase!

    # This method will get the classification for any new test samples we want to predict

    def predict(self, X):
        # Now we make a prediction with each of our trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Now we want to take the majority vote but be careful when doing this step
        # In our case, we will be using a built in numpy function to help us
        # The following switches from [1111 0000 1111] to [ (sample 1 preds) --> 101 (sample 2 preds) --> 101 101 101]
        # Esentially we want to group the predictions within the trees by sample and then proceed to get an accurate final prediction
        # By looking at the majority/most common prediction among each group of corres[onding samples
        tree_predictions = np.swapaxes(tree_predictions, 0, 1)

        # Now the following will get the most common label for each group of samples
        y_prediction = [_most_common_label(tree_prediction) for tree_prediction in tree_predictions]
        print(y_prediction)
        return np.array(y_prediction)
