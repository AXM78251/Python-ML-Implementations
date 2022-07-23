import numpy as np


# LDA = Linear Discrminant Analysis
# Dimensionality reduction technique similar to PCA
# Approach and implementation of PCA & LCA have a lot in commmon
# Goal = feature reduction
#   Project our datasets onto a lower dimensional space
#   Find a good class sepaaration
# Supervised technique --> we know the feature labels
# Want to find new axes such that the class separation is maximized
# LDA still want to have a good variance between the single features
#   Also interested in the axes that maximize the separation between multiple classes
# Within class scatter
#   Makes sure that our features within 1 class are good separated
# Between class scatter
#   Makes sure that all the classes are separated good
# Approach
#   Calculate the between class scatter matrix
#   Calculate the within class scatter matrix
#   Calculate the eigenvalues of the inverse within class scatter matrx * between class scatter matrix
#   Sort the eigenvectors according to their eigenvalues in decreasing order
#   Choose the first k eigenvectors, these will be the new k dimensions (linear discriminants)
#   Transform the original n dimensional data points into k dimensions
#       Projections with dot product

class LDA:

    # This method will preserve the arguments to the designated fields of the obj created

    def __init__(self, n_components):
        # Will dictate how many eigenvectors we will choose to form our new dimensions
        self.n_components = n_components

        # Here we will store the eigenvectors that we compute
        self.linear_discriminants = None

    # This method will fit training samples and training labels

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)
    # Y = 1D row vector of size M (training labels)

    # X = training samples
    # Y = training labels

    def fit(self, X, y):
        # Iris dataset --> 150 samples, 4 features
        # This will be used to get the number of features we have
        n_features = X.shape[1]

        # Will only return the unique values in our labels
        class_labels = np.unique(y)

        # Calculate the mean of all our samples
        mean_overall = np.mean(X, axis=0)

        # Initialize our two scatter matrices
        S_W = np.zeros((n_features, n_features))  # Size = 4 * 4
        S_B = np.zeros((n_features, n_features))  # Size = 4 * 4

        # Calculate our within class scatter matrix

        # Calculate the between class scatter matrix

        for c in class_labels:
            # First want to get only the samples of this class
            X_c = X[y == c]

            # Next get the mean of only the features present in the given class
            mean_c = np.mean(X_c, axis=0)

            # Size at the end should 4 * 4
            # Size of (X_c - mean_c) is num_samples_in_class * 4
            # Size of (X_c - mean_c).T is 4 * num_samples so to get a 4 * 4 matrix, we must multiply it by (X_c - mean_c)
            S_W += (X_c - mean_c).T.dot((X_c - mean_c))

            # Will return the number of samples in the given class
            # Must be careful, have to reshape vector
            num_samples_in_class = X_c.shape[0]

            # Calculate the difference between the mean of the class and the total mean
            # Only 1D, currently size (4,) but want it to be size (4, 1) so reshape it
            mean_difference = (mean_c - mean_overall).reshape(n_features, 1)
            S_B += num_samples_in_class * (mean_difference).dot(mean_difference.T)

        A = np.linalg.inv(S_W).dot(S_B)

        # Time to calculate eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(A)

        # One column is one eigenvector, make it so that 1 row = 1 eigenvector
        # Once again be careful, make sure to transpose our eigenvectors
        eigenvectors = eigenvectors.T

        # Sort eigenvectors
        # We want to sort our eigenvalues im decreasing order
        # We can use slicing for this
        # Neat little trick to reverse our list is featured below
        # Will return the indices of the eigenvalues in decreasing order
        indices = np.argsort(abs(eigenvalues))[::-1]

        # Sorts the eigenvalues in decreasing order
        eigenvalues = eigenvalues[indices]

        # Sorts the eigenvectors in decreasing order
        eigenvectors = eigenvectors[indices]

        # Store only the k first eigenvectors
        self.linear_discriminants = eigenvectors[0:self.n_components]

    # This method will take care of the new features we want to project
    def transform(self, X):
        # Project our data onto the new basis/dimensions via the dot product
        return np.dot(X, self.linear_discriminants.T)
