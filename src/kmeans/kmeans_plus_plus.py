# src/kmeans/kmeans_plus_plus.py

import numpy as np
from src.kmeans.distance import squared_euclidean_expansion
from src.kmeans.min_heap import MinHeap


def kmeans_plus_plus(X: np.ndarray, K: int, random_state: int = 42) -> np.ndarray:
    """
    K-Means++ initialization using expansion-based distance and custom Min-Heap.

    Parameters
    ----------
    X : ndarray of shape (N, F)
        Dataset
    K : int
        Number of clusters
    random_state : int
        Seed for reproducibility

    Returns
    -------
    centroids : ndarray of shape (K, F)
        Initialized centroids
    """
    rng = np.random.default_rng(random_state)
    N, F = X.shape

    centroids = np.empty((K, F))

    # 1️⃣ Choose first centroid uniformly at random
    first_idx = rng.integers(N)
    centroids[0] = X[first_idx]

    # 2️⃣ Initialize closest distances
    closest_dist_sq = squared_euclidean_expansion(X, centroids[0:1]).reshape(-1)

    # 3️⃣ Heap to track farthest points (use negative distance)
    heap = MinHeap()
    for i, d in enumerate(closest_dist_sq):
        heap.push((-d, i))

    # 4️⃣ Select remaining centroids
    for k in range(1, K):
        # Pick farthest point
        _, idx = heap.pop()
        centroids[k] = X[idx]

        # Update closest distances
        new_dist_sq = squared_euclidean_expansion(X, centroids[k:k+1]).reshape(-1)

        closest_dist_sq = np.minimum(closest_dist_sq, new_dist_sq)

        # Rebuild heap with updated distances
        heap = MinHeap()
        for i, d in enumerate(closest_dist_sq):
            heap.push((-d, i))

    return centroids
