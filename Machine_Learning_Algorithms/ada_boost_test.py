import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from ada_boost import Adaboost


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


# Load in breast cancer datasets from sklearn datasets
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# IMPORTANT
# Set all labels that are 0 at the moment to -1
# Adaboost needs labels as -1 or +1
y[y == 0] = -1

# Gets pur train and test splits
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Adaboost classification with 5 weak classifiers
clf = Adaboost(num_classifiers=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Calculates the accuracy
acc = accuracy(y_test, y_pred)
print("Accuracy:", acc)
