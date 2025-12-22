# tests/test_kmeans_plus_plus.py

import numpy as np
from src.kmeans.kmeans_plus_plus import kmeans_plus_plus


def test_kmeans_plus_plus_shape():
    X = np.random.rand(100, 5)
    K = 4

    C = kmeans_plus_plus(X, K, random_state=0)

    assert C.shape == (K, 5)


def test_kmeans_plus_plus_deterministic():
    X = np.random.rand(50, 3)

    C1 = kmeans_plus_plus(X, 3, random_state=123)
    C2 = kmeans_plus_plus(X, 3, random_state=123)

    assert np.allclose(C1, C2)
