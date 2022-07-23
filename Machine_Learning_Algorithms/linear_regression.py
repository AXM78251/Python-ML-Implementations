import numpy as np


# In regression, we want to predict continuous values
# Whereas in classification, we want to predict discrete values
#   Such as a class labeled 0 or 1, true or false, yes or no, etc.
# Use a linear function to predict the values such that the approximation is
#   y = wx + b (line equation),
#   w = weight (slope)
#   b = bias (shift along the y-axis in 2D case)
#   x = independent variable
#   y (prediction) = dependent variable

# Linear regression is categorized as supervised learning
# For the actual values, we need training samples hence why this is a supervised learning algorithm
# Used to predict the relationship between independent and dependent variables

# To find what a and b are, we define a cost function
# For linear regression, this is given to us by the mean squared error
# Mean Squared Error (MSE) function):
#   Square the error difference, sum over all data points, then divide this value by the total number of data points
#   Next step is to use our MSE function to change w & b such that the MSE settles at the minima -> gradient descent
# Gradient Descent
#   Method of updating w & b to reduce the MSE function
#   Start off with some initial w & b values and change these values iteratively to reduce the cost
#   Gradient Descent algorithm --> imagine a U shaped pit, and we are standing at topmost point in pit
#   Objective is to reach the bottom of the pit, but we can only take a discrete number of steps to reach the bottom
#   Smaller steps -> eventually reach the bottom, takes longer
#   Longer steps -> would reach sooner, chance that we overshoot and not end exactly at the bottom
#   The number of steps we take is the "learning rate"
# To update w & b, we take gradients of the MSE function
#   Take partial derivative with respect to w & b, requires some calculus
#   Partial derivatives are gradients used to update w & b
#   In the eq, we have alpha which is the learning rate and a hyperparameter we must specify
#   w = w - alpha * (partial derivative of w)
#   b = b - alpha * (partial derivative of b)

# Linear Regression derived from base regression
class LinearRegression():

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
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - Y))
            db = (1 / n_samples) * np.sum(y_predicted - Y)

            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    # This method will approximate value for any new test sample
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
