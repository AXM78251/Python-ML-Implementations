import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

cmap = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

iris = datasets.load_iris()
X, Y = iris.data, iris.target

# X_train = training samples
# X_test = test samples
# Y_train = training labels
# Y_test = test labels

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1234)

# Following statement will print shape of the training samples
# Prints a numpy nd array of shape 120 x 4

# 120 = number of samples
# 4 = number of features for each sample
# print(X_train.shape)

# Following statement will print the first sample
# print(X_train[0])

# Following statement will print shape of the training labels
# Prints a 1D row vector of 120
# For each of our training samples we have the label for it
# print(Y_train.shape)

# Prints out a 1D vector with only 1 row
# We have labels 0, 1, and 2 so this is a 3 class problem
# print(Y_train[0])

# Only plots the first 2 features so that we have a 2D case
# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c = Y, cmap = cmap, edgecolor = 'k', s = 20)
# plt.show()

# a = [1, 1, 1, 2, 2, 3, 4, 5, 6]
# from collections import Counter

# most_common = Counter(a).most_common(1)
# print(most_common)
# Will print out the following (a, b)
# a = the most common item
# b = how many times the item occurs

# Will print out most common item
# print(most_common[0][0])

from knn import KNN

# Create a classifier
clf = KNN(k=7)

# Call the fit method
clf.fit(X_train, Y_train)

# Get the predictions by calling clf.predict
predictions = clf.predict(X_test)

# Calculate the accuracy, defined by how many predictions are correctly classified
# Add 1 for every test sample that is correct and divide by the total amount of test samples we have
acc = np.sum(predictions == Y_test) / len(Y_test)
print(acc)
