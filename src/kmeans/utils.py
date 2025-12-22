# src/kmeans/utils.py

import numpy as np
from src.kmeans.kmeans import KMeans
from src.kmeans.hash_table import HashTable


def compute_wcss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Compute Within-Cluster Sum of Squares (WCSS).

    Parameters
    ----------
    X : ndarray of shape (N, F)
    labels : ndarray of shape (N,)
    centroids : ndarray of shape (K, F)

    Returns
    -------
    wcss : float
    """
    wcss = 0.0
    for k in range(len(centroids)):
        cluster_points = X[labels == k]
        if len(cluster_points) > 0:
            diff = cluster_points - centroids[k]
            wcss += np.sum(diff ** 2)
    return wcss


def elbow_method(
    X: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> HashTable:
    """
    Run Elbow Method for K-Means and store WCSS using custom Hash Table.

    Returns
    -------
    wcss_table : HashTable
        Maps K -> WCSS(K)
    """
    wcss_table = HashTable()

    for K in range(k_min, k_max + 1):
        km = KMeans(
            n_clusters=K,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
        )
        km.fit(X)
        wcss_table.put(K, km.inertia_)

    return wcss_table
