from collections import Counter

import numpy as np


# Supervised classification algorithm --> will require some training data
# At each node of the decision tree, we will either find the best split feature or best split value (and store them) to best fit our data accordingly w/o overfitting our data
# Entropy = measure for uncertainty, we will need to calculate this
# Will start off first by calculating the entropy, then we will split our data and then calculate the entropy for pur childs
# Then we will calculate how much information we gain through this split and this measure is called our information gain
# After getting our information gain, we perform a greedy search over all possible features and over all possible feature values
# Using this greedy search, we will then select the best feature and feature value to be updated to
# Train algorithm --> grow the tree
#   Start at top node and at each node select the best split based on the best information gain
#   Greedy search: Loop over all features and over all thresholds (all possible feature values)
#   Save the best split feature and split threshold at each node
#   Build the tree recursively
#   Apply some stopping criteria to stop growing --> maximum depth, minimum samples at node, no more class distribution in node, etc
#   When we have a leaf node, store the most common class label of this node
# Predict --> traverse the tree
#   Traverse the tree recursively
#   At each node look at the best split feature of the test feature vector x and go left or right depending on x[feature_idx] <= threshold
#   When we reach the leaf node, we return the stored most common class label

# First create our global entropy method
# Will take in a vector y of all our class labels
def entropy(y):
    hist = np.bincount(y)  # Will calculate number of occurences of all class labels
    ps = hist / len(y)  # Divide the previous value by the number of total samples
    return -np.sum([p * np.log2(p) for p in ps if
                    p > 0])  # Final part of pur entropy function where se sum over all va;ues and negate that va;ue


# Helper class node that will store all the information for pur node
class Node:
    # If we are in the middle of our tree, we must make sure to store the best split feature and the best split threshold
    # Also want to store the left and the right child trees --> will need them later
    # If we instead are at a leaf node, we want to store the most common class label
    # The asterisk within our parameter makes it so if we want to use our value parameter
    # Then we must use it as a keyword only parameter so later when creating our leaf node which only gets the value
    # Then we must also write value = some_value so that it is clearer that it is a leaf node
    # Make sure to store/preserve our values

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.right = right
        self.left = left
        self.value = value

    # Now we write a helper function that will help us determine if we are at a leaf node
    def is_leaf_node(self):
        return self.value is not None


# Now that we have finished our helper class for the nodes, we can start with the actual decision tree class
class DecisionTree:

    # Will of course get itself, but also make sure to pass in some stopping criteria
    # Also make sure to pass in some max depth for our tree to prevent overreaching
    # Also gets a parameter that is the number of features we are working with
    # As we specified before, we will do a greedy search over all the features
    # However we can also just loop over a subset of number of features and randomly select the substeps and this is one of the factors --> one of the reasons it is called random forest
    # Make sure to store/preserve our values

    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_feats

        # Must also specify a root so that we know where our tree starts
        self.root = None

    # Now we implement our fit function that will fit our training data and training labels

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # The number of rows is the number of samples and thr number of columns is the number of features
    # Y = 1D row vector of size M (training labels)

    # X = training data
    # y = training labels

    # This method will grow our tree

    def fit(self, X, y):

        # Now we will apply some safety check
        # Essentially if self.n_features is not specified, then we simply take the maximum number of features
        # Otherwuse we take the minimum of self.n_features and the second element of X.shape
        # This just makes we are never greater than the actual number of features
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])

        # This statement will call our helper function that will grow our tree and establish our root node
        self.root = self._grow_tree(X, y)

    # Now implement our grow tree helper function that also takes in our training data X and training labels y
    # As well as a depth variable to keep track of the depth of our tree
    def _grow_tree(self, X, y, depth=0):
        # Start off by getting the number of samples and features
        n_samples, n_features = X.shape

        # Also get the number of different labels
        n_labels = len(np.unique(y))

        # First, we apply our stopping criteria to see if we are at a leaf node
        # If so, then we set our leaf value to be the most common value among our training labels
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            # Now we create and return our leaf Node
            # Must explicitly declare value = ? when creating our leaf node
            return Node(value=leaf_value)

        # If we didn't have a leaf/meet the stopping criteria then we continue with the rest of the tree
        # First select our feature indices
        # Following statement will select a number between 0 and the number of features
        # The array we pass in should also be of size self.n_feats
        feat_indices = np.random.choice(n_features, self.n_features, replace=False)

        # Now we will perform our greedy search algorithm
        best_feature, best_threshold = self._best_criteria(X, y, feat_indices)

        # Now that we have selected the best feature/threshold splits, we will now split our tree
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)

        # Now with our left and right indices, we can continue growing our tree
        left_tree = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
        right_tree = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)

        # Now that we have recursively grown our trees, we return a new node in the middle
        # That uses the best feature/threshold values as well as the left amd right child trees
        return Node(best_feature, best_threshold, left_tree, right_tree)

    # Create another helper function called _best_criteria
    # This will br the part where we go over all the feature and feature values
    # To calculate the information gain
    def _best_criteria(self, X, y, feat_indices):
        best_gain = -1
        split_index, split_threshold = None, None
        for feat_idx in feat_indices:
            # For this part, we only want to select the column vector of the array X that is passed in
            # We are looking for all the samples at the specified index that is passed on
            # Features are column vectors which is exactly what we are doing here
            # Will be getting 1 column/feature vector at a time
            X_column = X[:, feat_idx]
            # Now we go over all possible thresholds
            # Don't want to check the same value twice
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                # Now we calculate the information gain
                # By calling a helper function that will calculate the information gain for us
                gain = self._information_gain(y, X_column, threshold)

                # If the calculated gain is greater than our previous gain
                # Then we must update our gain, our best split feature index as well as our best split threshold value
                if gain > best_gain:
                    best_gain = gain
                    split_index = feat_idx
                    split_threshold = threshold

            # At the end of pur greedy searh algorithm
            # We return the best split index and the best threshold value that we found
            return split_index, split_threshold

    # Helper function to calculate information gain
    def _information_gain(self, y, X_column, split_threshold):
        # Calculate parent entropy
        parent_entropy = entropy(y)

        # Generate a split in our tree
        left_indices, right_indices = self._split(X_column, split_threshold)

        # If either condition is met then we know our information gain will be 0 so we return 0
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        # Calculate weighted average of the child entropy
        # This will be the number of samples we are working with
        total_samples = len(y)

        # Now calculate the number of left samples and right samples we are working with
        right_samples = len(right_indices)
        left_samples = len(left_indices)

        # Now calculate our entropies
        entropy_left, entropy_right = entropy(y[left_indices]), entropy(y[right_indices])
        child_entropy = (left_samples / total_samples) * entropy_left + (right_samples / total_samples) * entropy_right

        # Finally return our information gain
        gain = parent_entropy - child_entropy
        return gain

    # This helper function will be in charge of generating a split in our tree
    # Will more or less be splitting our tree based on binary search tree rules
    # Every node in the left subtree is less than the root
    # Every node in the right subtree is of a value greater than the root

    def _split(self, X_column, split_threshold):
        # Remember to flatten our arrays such that they will be 1 dimensional for both
        # Will flatten it into a row major order 1 dimensional array
        left_indices = np.argwhere(X_column <= split_threshold).flatten()
        right_indices = np.argwhere(X_column > split_threshold).flatten()
        return left_indices, right_indices

    # This method will traverse our tree

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # Finally, we will create a helper function that will allow us to traverse the tree smoothly
    def _traverse_tree(self, x, node):
        # First check if we havereached a leaf node
        if node.is_leaf_node():
            return node.value

        # Otherwise we apply our question to see whether we traverse
        # The left subtree or the right subtree
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    # Create another helper function that will return to us the most common label
    # Function will take in a vector of the class labels
    def _most_common_label(self, y):
        # Create a counter obj
        # Will calculate all the number of occurences for all the y's
        counter = Counter(y)
        # Only interested in the first most common label
        # This returns a tuple containing the value as well as the number of occurences
        # We are only concerned with the value so we extract the 0th element
        _most_common = counter.most_common(1)[0][0]
        return _most_common
