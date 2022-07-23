import numpy as np


# Uses the boosting approacj
# Follows the boosting approach --> simple idea to combine multiple weak classifiers into one strong classifier
# First thing we need is a weak classifier --> weak learner (decision stump)
# Weak learner = very simple classifier, in case of the adaboost algorithm, we use a decision stump
# Decision stump = decision tree with only one split
# Essentially, we only look at 1 feature of our sample at only one threshold
# Based on if our feature value is greater or smaller than the threshold, then we say it is either class -1 or class +1
# Then we need the formula for the error
# During our first iteration the error = # of misclassifications / total # of samples
# The next time around when calculating the error, we want to take into account the weights
# So if a sample was misclassified, then we give it a higher weight for the next iteration
# Our formula then becomes the sum over the weights for all misclassifications for our error
# If the error > 0.5, just flip the decision (class decision) and the error so that the error becomes
#   error = 1 - error
# Weights are initially set to 1 / N for each sample
# We also have an update rule, to update our weights according
# If our new calculated weight = - 1, we have a misclassification, if it is + 1, then
# We have a correct classification
# Our weight updating formula essentially makes sure we have a higher impact for the next classifier
# We also need to calculate the performance for each classifier, also known as alpha
# Our final prediction will then just be sign of the sum over all predictions where
# We weigh each prediction with the performance of the classifier
# So alpha times the prediction H(X)
# So the better our classifier, the more impact it has for the final prediction
# The better the classifier, the more it points to the negative or positive side
# Training
#   Initialize weights for each sample = 1 / N
#       For t in T (the number of weak learners we want):
#           * train weak classifier (greedy search to find the best feature and threshold)
#           * Calculate the error for this decision stump --> flip error and decision if error > 0.5
#           * Calculate the classifier performance, alpha
#           * We need the predictions and alpha, then we can update our weights

# Helper class for our decision stump

class DecisionStump:

    # This will store our initial polarity as well as our feature index, the split threshold value
    # As well as our variable performance, alpha

    def __init__(self):
        # This will tell us if the sample should be classified as -1 or +1
        # For the given threshold --> if we want to look at the right or left side
        # This is needed because if we want to flip the error
        # We must also flip the polarity
        self.polarity = 1

        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    # Define a predict method for our decision stump
    # This method will predict any new samples that are passed in

    def predict(self, X):

        # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
        # X = training samples

        # What we want to do here...
        # Simply look at 1 feature of this sample
        # Then compare it with the threshold
        # If it's smaller it is -1, else it is +1
        num_samples = X.shape[0]

        # Then let's get only the features at the given feature index calculated earlier
        X_column = X[:, self.feature_idx]

        # Now make our predictions, by default we say it is 1
        # Size = n_samples
        predictions = np.ones(num_samples)

        # Must now check polarity (default case)
        if self.polarity == 1:
            # All predictions (where the feature vector) < threshold, then it is -1
            predictions[X_column < self.threshold] = -1

        else:
            # Do it for the other way around
            predictions[X_column > self.threshold] = -1

        return predictions


class Adaboost:

    def __init__(self, num_classifiers=5):
        self.num_classifiers = num_classifiers

    # This method will fit training samples and training labels

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # Y = 1D row vector of size M (training labels)

    # X = training samples
    # Y = training labels

    def fit(self, X, y):

        # First thing to do is get the shape of the vector
        n_samples, n_features = X.shape

        # Initialize our weights
        # This will create an array of size n_samples where
        # Each value will be equal to 1 / n_samples
        weights = np.full(n_samples, (1 / n_samples))

        # Iterate through all the classifiers to train them
        # First create a list to store all the classifiers
        self.classifiers = []
        for _ in range(self.num_classifiers):
            # Now here we do he greedy search algorithm
            # Iterate through all the features and all the thresholds
            # Our classifier will be the Decision Stump
            clf = DecisionStump()

            # Find best split feature value and split threshold value
            # Where the error is then the minimum
            # In the beginning, this is some very large quantity
            min_error = float('inf')

            # Now iterate over all the features
            for feature_idx in range(n_features):
                # Want to get all the samples only at the
                # Specified feature index
                # Features are column vectors which is exactly what we are doing here
                # Will be getting 1 column/feature vector at a time
                X_column = X[:, feature_idx]

                # Now we want to get only the unique values/thresholds from the feature vector
                thresholds = np.unique(X_column)

                # Now that we have this, iterate over all our threshold values
                for threshold in thresholds:
                    # We want to predict with the polarity 1 first
                    # Then go on to calculate the errors
                    polarity = 1
                    predictions = np.ones(n_samples)
                    # All predictions (where the feature vector) < threshold, then it is -1
                    predictions[X_column < threshold] = -1

                    # Now that we have predicted all the samples
                    # Calculate the error
                    # Error is the sum over the weights of the misclasdified samples
                    misclassified = weights[y != predictions]
                    error = sum(misclassified)

                    # Check if error and polarity must be flipped
                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    # Check whether we must update our min_error
                    # If this is the case, we must also
                    # Store the polarity as well as the threshold and the feature index
                    if error < min_error:
                        min_error = error
                        clf.polarity = polarity
                        clf.threshold = threshold
                        clf.feature_idx = feature_idx

            # Done with both for loops
            # Now, calculate the performance (alpha)
            # Use a small epsilon value so we don't divide by 0
            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1 - error) / (error + EPS))

            # Now update our weights (also need predictions for this)

            # The following are our predictions which is calculated
            # From the predict function inside the DecisionStump class
            predictions = clf.predict(X)

            weights *= np.exp(-clf.alpha * y * predictions)

            # Now we want to normalize them
            # Divide by the sum over these weights
            weights /= np.sum(weights)

            # Finally store this classifier
            self.classifiers.append(clf)

    def predict(self, X):
        # Do this for every classifier we have
        # For every classifier we multiply the respective alpha and preductions
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.classifiers]

        # We then sum over all these values
        y_pred = np.sum(clf_preds, axis=0)

        # Lastly, we get the sign of this value
        # This sign is what we will return
        y_pred = np.sign(y_pred)
        return y_pred
