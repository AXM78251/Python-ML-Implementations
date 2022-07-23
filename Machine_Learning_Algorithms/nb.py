import numpy as np


# Algorithm is based off Bayes Theorem
# Called Naive Bayes because we make the assumption that
# All features are mutually independent
# P(y|X) = posterior probability
# P(x|y) = class conditional probability
# P(y) = prior probability of y
# P(X) = prior probability of X
# Now we want to make a clssification
# Given the posterior probability, we want to select the class with the highest probability
# Choose y to be the argmax of y of the posterior probability P(y|X)
# Since we are only concerned with y, we don't need P(X)
# This leaves us with y being the argmax of y of the conditional probability P(x|y) for all x in X multiplied by the prior probability of y P(Y)
# Since all the probability values when multiplied my each other may get smaller and smaller, this may result in overflow
# To work around this, we can apply the log function using logarithmic rules, effectively changing the * to +
# Prior probability P(y) is just the frequency
# Class conditional probability, P(x|y), can be modeled using a gaussian distribution

class NaiveBayes:

    # Does not need init method, go on to fit method right away

    # This method will fit training data and training labels

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # Y = 1D row vector of size M (training labels)

    # X = training data
    # y = training labels
    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Below statement will find the unique elements of an array, in this case y
        self._classes = np.unique(y)

        # Below statement will get the number of unique classes
        n_classes = len(self._classes)

        # init mean, variance, priors
        # For each class, we need means for each feature
        # The following are numpy nd vectors of size M x N where
        # M = n_classes
        # N = n_features
        # dtype = desired data-type for the array, in this case we are dealing wuth 64 bit floating point values
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._variance = np.zeros((n_classes, n_features), dtype=np.float64)

        # For each class, we want 1 prior
        # 1D vector of size n_classes
        # dtype = desired data-type for the array, in this case we are dealing wuth 64 bit floating point values
        self._priors = np.zeros(n_classes, dtype=np.float64)

        # Only want the samples that have the class c as labels
        # Iterates through every single unique class
        for c in self._classes:
            X_c = X[c == y]
            # Find the mean if the class and put it back into the respective idx of said class
            self._mean[c, :] = X_c.mean(axis=0)
            # Find the variance if the class and put it back into the respective idx of said class
            self._variance[c, :] = X_c.var(axis=0)

            # Prior prob that this class will occur is equal to the frequency of this class in the training samples
            # Will get the number of samples with this label and then divide by the number of total samples
            # Will represent how frequently this class c is occuring
            self._priors[c] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_predict = [self._predict(x) for x in X]
        return y_predict

    # Will get one sample at a time
    # Calculate the posterior probability and calculate the class cinditional and the prior for each one
    # Finally, will choose the class with the highest probability
    def _predict(self, x):
        posteriors = []

        # The following will get the index and class label associated with said index
        for idx, c in enumerate(self._classes):
            # Get the prior probability first
            # Make sure to apply the log function as well
            prior = np.log(self._priors[idx])

            # We will now need the gaussian function
            # Our class conditional is the pdf of our given index and x
            # Make sure to use our log function
            # Sum all of them up as well
            class_conditional = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + class_conditional

            # Append them to our posteriors list
            # Now apply the argmax of this, choose the class with the highest probability
            posteriors.append(posterior)

        # Will return the class with the highest probability, and therefore the returned value will be our classification
        return self._classes[np.argmax(posteriors)]

    # Probability density function, will be used to get our class conditional probability
    def _pdf(self, class_idx, x):
        # Will get the mean of the given class index
        mean = self._mean[class_idx]

        # Will get the variance of the given class index
        variance = self._variance[class_idx]

        numerator = np.exp(- (x - mean) ** 2 / (2 * variance))
        denominator = np.sqrt(2 * np.pi * variance)
        return numerator / denominator

        # This _pdf is our probability density function
