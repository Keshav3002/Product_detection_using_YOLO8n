from sklearn.cluster import KMeans
import numpy as np

class ProductGrouper:
    def __init__(self, n_clusters=10):
        """Initialize the product grouper with the number of clusters."""
        self.n_clusters = n_clusters

    def group(self, feature_vectors):
        """Group products based on their feature vectors using k-means clustering."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=0)
        clusters = kmeans.fit_predict(feature_vectors)
        return clusters
