import matplotlib.pyplot as plt
import numpy as np

# Goal = cluster a data set into k different clusters
# Data set is unlabeled --> unsupervised learning
# Each sample should be assigned to the cluster with the nearest mean
# Iterative optimization technique
#   Initialize cluster centers (e.g. randomly)
#   Repeat until converged:
#       Update cluster labels: Assign points to the nearest cluster center (centroids)
#       Update cluster centers (centroids): Set center to the mean of each cluster
# Euclidean distance
#   Will need this to get the distance between two feature vectors
#

# Don't need this, but we could use this to reproduce data later with same results
np.random.seed(42)


# Make a global euclidean distance function first
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


class KMeans:

    # This method will preserve the designated arguments passed in to the designated fields of the obj created
    # K = number of clusters we will have
    # Max_iters = the maximum number of iterations we want to do for our optimization
    # Plot_steps = false (not required) but we will implement this in order to plot the different steps we are in

    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps

        # Now create our empty clusters and centroids
        # Our clusters are a list of lists
        # Essentially it is a list of sample indices for each cluster
        # So for each cluster we initialize an empty list
        self.clusters = [[] for _ in range(self.K)]

        # Store the feature vectors of the mean feature vector or each cluster
        # Mean feature vector for each cluster
        self.centroids = []

        # Clusters will only hold indices wheras the centroids list will actually hold samples

    # For an unsupervised learning technique, we just implement the predict method
    # Unsupervised because we have no training data to train our model

    def predict(self, X):

        # First store our data
        self.X = X

        # Store the dimensions of our data
        self.n_samples, self.n_features = X.shape

        # Initialize our centroids (randomized)
        # Replace = false because we don't want to pick the same indices twice
        # The result will be an array of size self.K and for each entry
        # It will pick a random choice between 0 and self.n_samples
        random_sample_indices = np.random.choice(self.n_samples, self.K, replace=False)

        # Now we use the indices to get the random sample to be our centroid
        self.centroids = [self.X[idx] for idx in random_sample_indices]

        # Do our optimization
        for _ in range(self.max_iters):
            # In this for loop, we will plot after we update our clusters and our centroids

            # First update our clusters
            # Will call our helper function that will aid in creating our clusters for us
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # Then update the centroids, but first store the old centroids to check for convergence later
            centroids_old = self.centroids

            # We will call another helper function
            # This will assign the mean value of the clusters to the centroids
            # So for each cluster, we now calculate the mean
            self.centroids = self._get_centroids(self.clusters)
            if self.plot_steps:
                self.plot()

            # Then check for convergence, if convergence achieved then break
            # We will call another helper function to check for convergence
            # Will simply check the distances between each old and new centroids for all the centroids
            # Will check if the difference is 0
            if self._is_converged(centroids_old, self.centroids):
                break

        # At the end, we want to classify the samples as the index of their clusters
        # Return cluster labels
        # For each sample, we will get the label of the cluster it was assigned to
        return self._get_cluster_labels(self.clusters)

    # Here, we will assign the samples to the closest centroids to create our clusters
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        # This enumerate function will give us the current index and current sample associated with said index
        for idx, sample in enumerate(self.X):
            # Now we want to get the closest centroid, we want to have the index of this
            centroid_idx = self._closest_centroid(sample, centroids)
            # After getting the closest centroid, we then append the current sample index into the closest cluster
            clusters[centroid_idx].append(idx)
        # After iterating through, we then return our fully populated clusters
        return clusters

    # This helper function will return to us the closest centroid
    def _closest_centroid(self, sample, centroids):
        # Here we calculate the distances of the current sample to each centroid
        # Then  we want to get the index of the centroid which has the closest distance
        distances = [euclidean_distance(sample, point) for point in centroids]

        # Once we get all our distances, we then want to get the index with the minimum distance
        closest_index = np.argmin(distances)

        # Return the closest index
        return closest_index

    def _get_centroids(self, clusters):
        # Initialize our centroids with 0 at the beginning
        # For each cluster, we will store the feature vector
        # Hence why we must have the dimensions self.K x self.n_features
        # Now we iterate over the clusters
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            # Calculate the cluster mean
            # Clusters equal a list of lists
            # So when we get yhe current from the enumerate function
            # We get a list of the indices thatare in the current cluster
            # So therefore, we must call self.X on those indices to get the actual samples in the current cluster
            # Then we calculate the mean of all these samples
            cluster_mean = np.mean(self.X[cluster], axis=0)

            # Now we assign our mean to the current centroid
            centroids[cluster_idx] = cluster_mean

        # Now we can return our new/fully updated centroid means
        return centroids

    # Will simply check the distances between each old and new centroids for all the centroids
    # Will check if the difference is 0
    def _is_converged(self, centroids_old, centroids_new):
        # For each cluster, it will look at the old and new centroid vector
        # Will calculate the euclidean distance and store it in the list
        distances = [euclidean_distance(centroids_old[i], centroids_new[i]) for i in range(self.K)]

        # Finally, we can return the sum of distances
        # If the sum of distances equals 0 then we have no more change in our centroids
        # At this point, we are effectively done
        return sum(distances) == 0

    # For each sample, we will get the label of the cluster it was assigned to
    def _get_cluster_labels(self, clusters):
        # Create an empty list that will contain our cluster labels
        # For each sample, we want to return which is the cluster it was assigned to
        # Be careful because these labels are not the actual labels of our data
        # Because we don't know them, so this is just the index of the cluster it was assigned to

        labels = np.empty(self.n_samples)

        # Now we iterate over our clusters
        for cluster_idx, cluster in enumerate(clusters):
            # Now we iterate over all the samples in the current cluster
            for sample_idx in cluster:
                # Assign the label of the current sample index to be the current cluster index
                labels[sample_idx] = cluster_idx

        return labels

    # This final function will be our plotting function, not needed but recommended for visualization purposes
    def plot(self):
        # Simply want to plot the data (to which cluster it belongs)
        # As well as the centroids
        # Create our figure of size 12 x 8
        fig, ax = plt.subplots(figsize=(12, 8))

        # Next iterate over our clusters
        for i, index in enumerate(self.clusters):
            # Now we get the current point
            point = self.X[index].T

            # Scatter the point, make sure to unpack the point
            # Use the * denotation
            # This will plot all the points
            # For each cluster, it will use a different color
            ax.scatter(*point)

        # Now we will iterate over/plot all the centroids
        for point in self.centroids:
            ax.scatter(*point, marker="X", color="black", linewidth=2)

        plt.show()
