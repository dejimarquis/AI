import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Data import Data




# ================= Helper Function for Plotting ====================
def plot_centroids(centroids, marker='o', scale=1):
    for c, centroid in enumerate(centroids):
        plt.plot([centroid[0]], [centroid[1]], color=cm(1. * c / K), marker=marker,
                 markerfacecolor='w', markersize=10*scale, zorder=scale)

# ===================================================================


class kmeans():
    def __init__(self, K):
        self.K = K      # Set the number of clusters

        np.random.seed(1234)
        self.centroids = np.random.rand(K, 2) # Initialize the position for those cluster centroids

        # Plot initial centroids with 'o'
        plot_centroids(self.centroids, marker='o')

    def fit(self, X, iterations=10):
        """
        :param X: Input data, shape: (N,2)
        :param iterations: Maximum number of iterations, Integer scalar
        :return: None
        """
        self.C = -np.ones(np.shape(X)[0],dtype=int) # Initializing which center does each sample belong to
        for iter in range(iterations):
            self.assign_clusters(X, self.C, self.centroids)
            self.update_centroids(X, self.C, self.centroids)
        return 0

    def update_centroids(self, X, C, centroids):
        """
        :param X: Input data, shape: (N,2)
        :param C: Assigned clusters for each sample, shape: (N,)
        :param centroids: Current centroid positions, shape: (K,2)
        :return: Update positions of centroids, shape: (K,2)

        Recompute centroids
        """

        for i in range(centroids.shape[0]):
            points = [X[j] for j in range(X.shape[0]) if C[j] == i]
            centroids[i] = np.mean(points, axis=0)

        return centroids

    def assign_clusters(self, X, C, centroids):
        """
        :param X: Input data, shape: (N,2)
        :param C: Assigned clusters for each sample, shape: (N,)
        :param centroids: Current centroid positions, shape: (K,2)
        :return: New assigned clusters for each sample, shape: (N,)

        Assign data points to clusters
        """
        print(centroids)
        for i in range(X.shape[0]):
            C[i] = np.argmin(np.sum(np.square(centroids-X[i]), axis=1))

        return C

    def get_clusters(self):
        """
        :return: assigned clusters and centroid locaitons, shape: (N,), shape: (K,2)
        """
        return self.C, self.centroids


if __name__ == '__main__':
    cm = plt.get_cmap('gist_rainbow') # Color map for plotting

    # Get data
    data = Data()
    X = data.get_kmeans_data()

    # Compute clusters for different number of centroids
    for K in [2, 3, 5]:
        # K-means clustering
        model = kmeans(K)
        model.fit(X)
        C, centroids = model.get_clusters()

        # Plot final computed centroids with '*'
        plot_centroids(centroids, marker='*', scale=2)

        # Plot the sample points, with their color representing the cluster they belong to
        for i, x in enumerate(X):
            plt.plot([x[0]], [x[1]], 'o', color=cm(1. * C[i] / K), zorder=0)
        plt.axis('square')
        plt.axis([0, 1, 0, 1])
        plt.savefig('figures/Q3_' + str(K) + '.png')
        plt.close()