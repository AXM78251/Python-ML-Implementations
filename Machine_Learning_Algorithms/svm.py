import numpy as np


# Support vector machine, follows idea to use a linear model and
# To find a linear decision boundary, called a hyperplane, that best separates our data
# For us, the hyperplane is the one that represents the largest separation or the largest margin in between the two classes
# So we choose the hyperplane so that the distance from it to the nearest data pt on each side is maximized
# For the math, we use a linear model -> w * x - b = 0, function must also satisfy condition that w * x - b >= 1 for our class 1 and w * x - b <= -1 for our class -1
# Putting it together, we get y(w * x - b) >= 1 where y = class label and this is the condition we must satisfy
# Must find our weights, w, and bias, b and for this we use a cost function and then apply a gradient descent
# In our case, our cost function will be the hinge loss function
# Hinge loss will be 0 if data is correctly classified and larger than 1
# Otherwise, then we have a linear function where the further away we are from the decision boundary line, the higher our loss is
# The other part of our cost function involves maximizing the margin between both of our classes
# The margin for us is 2 / magnitude of w which is our weights vector, so to maximize this value, we must minimize the magnitude of our weights vector
# We also add a regularization parameter to our cost function and we do this to balance the margin maximization and loss
# Now that we have this, we can them take partial derivatives with respect to the weights and bias to find the gradients and we then use these gradients to update our weights and bias

class SVM:

    # Instantiate all the fields for the object that was just created
    # These fields include the weights and bias which we will find later
    # Also includes the learning rate, alpha, the lambda parameter for our regularization function, and the number of iterations used in our gradient descent step

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    # This method will fit training data and training labels

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # The number of rows is the number of samples and thr number of columns is the number of features
    # Y = 1D row vector of size M (training labels)

    # X = training data
    # y = training labels
    def fit(self, X, y):

        # This statement will convert all the y_train data to -1 or 1 depending on whether the data point is <= 0
        y_ = np.where(y <= 0, -1, 1)

        # Now we get the number of samples and number of features
        n_samples, n_features = X.shape

        # Initialize our weights and bias, our weights are really a vector of weights and our bias is a unit quantity
        # For each feature component, we put in a 0 for our weight component
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Now we can start our gradient descent
        for _ in range(self.n_iters):
            # Iterate over our training samples
            # The enumerate function will gives us the current index and the current sample associated with said index
            # And this will allow us to iterate over our training samples
            for idx, x_i in enumerate(X):
                # First we must determine whether we satisfy our condtion where y_i * f(x) >= 1
                # Below statement must be true for condition to be true
                condition = y_[idx] * (np.dot(x_i, self.weights) - self.bias)
                # Based on whether our condition is met, we will then update our weights and bias accordingly
                if condition >= 1:
                    self.weights = self.weights - self.lr * (2 * self.lambda_param * self.weights)
                else:
                    self.weights = self.weights - self.lr * (2 * self.lambda_param - np.dot(x_i, y_[idx]))
                    self.bias = self.bias - self.lr * y_[idx]

    # This method will predict the output of any new data point from X_test
    def predict(self, X):
        linear_output = np.dot(X, self.weights) - self.bias
        return np.sign(linear_output)
