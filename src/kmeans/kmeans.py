# src/kmeans/kmeans.py

import numpy as np
from src.kmeans.distance import squared_euclidean_expansion
from src.kmeans.kmeans_plus_plus import kmeans_plus_plus
from src.kmeans.min_heap import MinHeap


class KMeans:
    """
    High-performance K-Means clustering (Lloyd's Algorithm).
    """

    def __init__(
        self,
        n_clusters: int,
        max_iter: int = 300,
        tol: float = 1e-6,
        random_state: int = 42,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X: np.ndarray):
        """
        Fit K-Means to the dataset X.
        """
        N, F = X.shape

        # 1️⃣ Initialize centroids using K-Means++
        centroids = kmeans_plus_plus(
            X, self.n_clusters, random_state=self.random_state
        )

        for iteration in range(self.max_iter):
            old_centroids = centroids.copy()

            # 2️⃣ Assignment Step (vectorized)
            dist = squared_euclidean_expansion(X, centroids)
            labels = np.argmin(dist, axis=1)

            # Cache nearest distances (for empty cluster handling)
            closest_dist_sq = dist[np.arange(N), labels]

            # 3️⃣ Update Step
            for k in range(self.n_clusters):
                cluster_points = X[labels == k]

                if len(cluster_points) == 0:
                    # Empty cluster handling using Min-Heap
                    heap = MinHeap()
                    for i, d in enumerate(closest_dist_sq):
                        heap.push((-d, i))  # max-distance via negative key

                    _, idx = heap.pop()
                    centroids[k] = X[idx]
                else:
                    centroids[k] = np.mean(cluster_points, axis=0)

            # 4️⃣ Convergence Check
            if np.allclose(old_centroids, centroids, rtol=self.tol, atol=self.tol):
                break

        # Final assignments and inertia
        final_dist = squared_euclidean_expansion(X, centroids)
        labels = np.argmin(final_dist, axis=1)
        inertia = np.sum(final_dist[np.arange(N), labels])

        self.centroids_ = centroids
        self.labels_ = labels
        self.inertia_ = inertia

        return self

    def fit_predict(self, X: np.ndarray):
        self.fit(X)
        return self.labels_
