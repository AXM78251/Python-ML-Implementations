import numpy as np


# Logistic regression predicts whether something is true or false instead of predicting something continuous
# Mainly used for classification, if probability of one item is > 50%, we will classify it as one class
#   Otherwise, it's classified as the other
# Logistic Regression - We want a probability
# In order to model probability, we apply the sigmoid function to our linear model
# Will output a probability between 0 and 1
# With this function, we can model the probability of our data
# Similar to linear regression, we must come up with the parameters w & b
# w = weight
# b = bias
# We utilize the gradient descent algorithm to find these values as well
# Gradient Descent
#   Method of updating w & b to reduce the MSE function
#   Start off with some initial w & b values and change these values iteratively to reduce the cost
#   Gradient Descent algorithm --> imagine a U shaped pit, and we are standing at topmost point in pit
#   Objective is to reach the bottom of the pit, but we can only take a discrete number of steps to reach the bottom
#   Smaller steps -> eventually reach the bottom, takes longer
#   Longer steps -> would reach sooner, chance that we overshoot and not end exactly at the bottom
#   The number of steps we take is the "learning rate"
# To update w & b, we take gradients of the cost function
#   Take partial derivative with respect to w & b, requires some calculus
#   Partial derivatives are gradients used to update w & b
#   In the eq, we have alpha which is the learning rate and a hyperparameter we must specify
#   w = w - alpha * (partial derivative of w)
#   b = b - alpha * (partial derivative of b)


class LogisticRegression():

    # This method will preserve the learning rate and number of iterations to the designated fields of the obj created

    # lr = learning rate (usually very small value)
    # n_iters = number of iterations (steps) we use in our gradient descent algorithm
    def __init__(self, lr, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters

        # Come up with some initial value for the weight and bias
        self.weights = None
        self.bias = None

    # This method will fit training samples and training labels
    # This will involve the training step and gradient descent

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # Y = 1D row vector of size M (training labels)

    # X = training samples
    # Y = training labels
    def fit(self, X, Y):
        # initial parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.weights) + self.bias
            # Below statement applies sigmoid function to linear model
            y_predicted = self._sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - Y))
            db = (1 / n_samples) * np.sum(y_predicted - Y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    # This method will get the classification for any new test samples we want to predict
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    # Helper method where we will apply the sigmoid function
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
