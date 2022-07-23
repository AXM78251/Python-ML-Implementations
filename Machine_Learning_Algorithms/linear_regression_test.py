import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], y, color="b", marker="o", s=30)
# plt.show()

# Our X is an nd array of size 80 x 1
# 80 = number of samples
# 1 = number of features per sample
# print(X_train.shape)

# Our training labels is a 1D row vector of size 80
# For each training sample, we have 1 value
# print(y_train.shape)

from linear_regression import LinearRegression

# Create a new instance of LinearRegression
regressor = LinearRegression(lr=0.01)

# Call the fit function on the training samples and training labels
regressor.fit(X_train, y_train)

# Call the predicted method on the testing samples
predicted = regressor.predict(X_test)


def mse(y_true, y_predicted):
    return np.mean((y_true - y_predicted) ** 2)


mse_value = mse(y_test, predicted)
print(mse_value)

y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()
