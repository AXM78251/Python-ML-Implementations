import numpy as np


# Perceptron only works for linearly separable classes
# For further improvements, we can change the activation function --> such as using a sigmoid function and applying a gradient descent rather than the perceptron rule to update our weights
# Perceptron can be seen as one single unit of an artificial neural network
# Perceptron is a simplified model of a biological neuron and it simulates the behavior of only one cell
# Our cell gets an input so it gets input signals and they are weighted and summed up
# If the whole input signal then reaches a certain threshold, our cell fires a signal and delivers an output
# In our case, it either fires a 1 or 0
# Modeling this mathematically, it looks like the following:
# We have our input features and they are multiplied with some weights and then summed up and then we apply an activation function and get our output class
# Inputs can be features such as salary, marriage status, age, past credit profile, etc.
# Output is a boolean value: True or False, Yes or No, an example can be yes to approve a loan or no to reject a loan
# Our implemenation will follow a linear approach y = w * x + b where
# w = vector of real-valued weights
# x = vector of input x values
# b = bias (an element that adjusts the boundary away from origin without any dependence on the input value)
# Approximation ÿ = g(w * x + b), where we first get the result of the linear function and apply the activation function to said result
# Perceptron update rules
#   w = w + Δw
#   Δw = α * (actual label - ÿ) * training sample (x)
#   α = learning rate between 0 & 1, scaling factor
# We look at each training sample and apply the update rule
# We will also iterate through this process a designated amount of times until we get our final weights

class Perceptron:

    # This init method will store the learning rate and number of iterations to the designated fields of the object created

    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step_func
        self.weights = None
        self.bias = None

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # Y = 1D row vector of size M (training labels)

    # This method will fit all the training samples and training labels
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init weights, bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        y_ = np.array([1 if i > 0 else 0 for i in y])

        # Now we can start training our data
        # This will involve our update step and will consist of 2 for loops
        # 2 for loops because we want to apply the update rule for each training sample
        for _ in range(self.n_iters):
            # We want to iterate over the training samples in inner for loop
            # So the enumerate function will give us the index and also the current sample of said index
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                # Below statement uses the activation function for only one sample
                y_predicted = self.activation_func(linear_output)

                update = self.lr * (y_[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    # This method will predict/xlassify any new test samples
    def predict(self, x):
        linear_output = np.dot(x, self.weights) + self.bias
        # Below statement uses the activation function for multiple samples
        y_predicted = self.activation_func(linear_output)
        return y_predicted

    # This is the activation function which is simply the unit step function
    def _unit_step_func(self, x):

        # This will only work for one single sample, but later we see that
        # We want to apply the activation samples in the predict method for all the test samples
        # So we want to apply this for an nd array as well
        # return 1 if x >= 0 else 0
        # Instead do np.where which will work for one single sample but also for multiple samples in one vector

        return np.where(x >= 0, 1, 0)
