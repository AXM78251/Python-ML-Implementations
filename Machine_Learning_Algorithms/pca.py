import numpy as np


# PCA = principal component analysis
# PCA is a nice tool to get linearly independent features
# And also to reduce the dimensionality of our dataset
# Goal = find a new set of dimensions such that all the dimensions are orthogonal and hence linearly independent
#   Must also be ranked according to the variance of data along them
# This means we want to find a transformation such that the
# Transformed features are linearly independent
# The dimemnsionality can then be reduced by taking only the dimensions with the highest importance
# These newly found dimenensions should minimize the projection error
# The projected points should also have a maximum spread --> which means the maximum variance
# Must find the variance along with the covariance of our training dataand training labels
# Afterwards, we must then find the eigenvectors and eigenvalues of our covariance matrix
# The eigenvectors then point in the direction of the maximum variance
# The corresponding eigenvalues indivate the importance of its corresponding eigenvector
# Approach
# Eigenvectors will be orthogonal to each other
#   Subtract the mean value from X (training data)
#   Calculate the covariance of X --> Cov(X,X)
#   Calculate the eigenvectors and eigenvalues of the covariance matrix
#   Sort the eigenvectors according to their eigenvalues in decreasing order
#   Choose the first k eigenvectors that will be the new k dimensions
#   Transform the original n dimensional data points into k dimensions (= Projections with dot product)

class PCA:

    # This method will preserve the designated arguments passed in to the designated fields of the obj created
    # Make sure to specify the number of components we want to keep

    def __init__(self, n_components):
        self.n_components = n_components

        # Also want to find the eigenvectors
        self.components = None

        # Also make sure to store the mean
        self.mean = None

    # This method will fit training samples

    # X = numpy nd vector of size M x N, M = number of samples, N = number of features for each sample (training samples)

    # X = training samples

    def fit(self, X):
        # Find the mean of our data along the first axis
        self.mean = np.mean(X, axis=0)

        # Subtract our mean
        X = X - self.mean

        # Calculate the covariance matrix using a built in numpy function
        # Be careful because X is a numpy nd array where 1 row = 1 sample, 1 column = 1 feature
        # So we must transpose X so that we get the values of the corresponding feature together
        cov = np.cov(X.T)

        # Calculate our eigenvectors and eigenvalues using a built-in numpy function
        # One column is one eigenvector, make it so that 1 row = 1 eigenvector
        # Once again be careful, make sure to transpose our eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T

        # Sort eigenvectors
        # We want to sort our eigenvalues im decreasing order
        # We can use slicing for this
        # Neat little trick to reverse our list is featured below
        indices = np.argsort(eigenvalues)[::-1]

        # Following statement sorts the eigenvalues in decreasing order
        eigenvalues = eigenvalues[indices]

        # Now that we have the indices of the sorted eigenvalues in decreasing order, we can now
        # Store only the first n eigenvectors
        # Following statement sorts the eigenvectors in decreasing order
        eigenvectors = eigenvectors[indices]
        # Finally, we store the most important eigenvectors as dictated by our n_components field
        self.components = eigenvectors[0:self.n_components]

    # For this algorithm we don't use a predict method
    # Instead we use a transform method
    # This method will transform our data once we have fitted it
    # Transform via projection

    def transform(self, X):
        # Project our data
        # Don't forget to also subtract the mean here
        X = X - self.mean

        # Now we can project it and return it
        # Be careful --> make sure to transpose our components matrix
        return np.dot(X, self.components.T)
