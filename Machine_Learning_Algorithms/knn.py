from collections import Counter

import numpy as np


def euclidean_distances(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Knn is a supervised classification algorithm that
# Classifies new data points based on the nearest k nearest data points

# To calculate the distance between the new data point(s) and the k nearest data point(s)
# We use the Euclidean distance formula given by d = sqrt( (x2 - x1)^2 + (y2 - y1)^2 )

# In a more general case, the formula is just extended over an n-space area

# Remember....in a python class, first argument of every single method created will be self
# Self is a variable that points to the instance of the class that we're currently working with

# Training data = initial data used to train machine learning

class KNN:

    # This method will preserve the constant value k to the field "k" of the instance/obj we have just created

    # Self = instance of the class we're currently working with
    # K = number of nearest neighbors that we want to consider
    def __init__(self, k=3):
        self.k = k;

    # This method follows convention of other machine learning libraries
    # This method will fit the training samples and some training labels
    # Store the training samples and training labels

    # X = training samples
    # Y = training labels
    def fit(self, X, Y):
        self.X_train = X
        self.Y_train = Y

    # This method will predict new samples
    # Can get multiple samples
    # Will call the predict helper for all samples "x" in "X"

    # X = new samples
    # Returns a numpy array
    def predict(self, X):
        predicted_labels = [self.predict_helper(x) for x in X]
        return np.array(predicted_labels)

    # Helper method for the above predict method that will
    # Only take 1 sample at a time
    def predict_helper(self, x):
        # Compute distances of new sample "x" to all the training samples
        distances = [euclidean_distances(x, x_train) for x_train in self.X_train]

        # Get k nearest samples/neighbors, labels
        # Following statement will sort the distances from least to greatest, will return the corresponding index of the values
        # Will hold the indices of the k nearest samples
        k_indices = np.argsort(distances)[0:self.k]

        # Will get the labels of our nearest neighbors
        k_nearest_labels = [self.Y_train[i] for i in k_indices]

        # Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)

        # Will print out most common item
        return most_common[0][0]
