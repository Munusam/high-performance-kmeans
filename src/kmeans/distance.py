import numpy as np


def squared_euclidean_difference(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Difference-based vectorized squared Euclidean distance.

    Computes:
        ||x - c||^2 = sum((x - c)^2)

    Parameters
    ----------
    X : ndarray of shape (N, F)
        Data points
    C : ndarray of shape (K, F)
        Centroids

    Returns
    -------
    dist : ndarray of shape (N, K)
        Squared distances
    """
    # Broadcasting: (N, 1, F) - (1, K, F) -> (N, K, F)
    diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]
    return np.sum(diff ** 2, axis=2)


def squared_euclidean_expansion(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """
    Expansion-based vectorized squared Euclidean distance.

    Computes:
        ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x · c

    Parameters
    ----------
    X : ndarray of shape (N, F)
        Data points
    C : ndarray of shape (K, F)
        Centroids

    Returns
    -------
    dist : ndarray of shape (N, K)
        Squared distances
    """
    # ||x||^2 -> (N, 1)
    x_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)

    # ||c||^2 -> (1, K)
    c_norm = np.sum(C ** 2, axis=1).reshape(1, -1)

    # x · c^T -> (N, K)
    cross_term = X @ C.T

    return  x_norm + c_norm - 2.0 * cross_term
