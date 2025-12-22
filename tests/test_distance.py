# tests/test_distance.py

import numpy as np
from src.kmeans.distance import (
    squared_euclidean_difference,
    squared_euclidean_expansion
)


def test_distance_equivalence():
    np.random.seed(42)

    X = np.random.randn(100, 5)
    C = np.random.randn(7, 5)

    d1 = squared_euclidean_difference(X, C)
    d2 = squared_euclidean_expansion(X, C)

    assert np.allclose(d1, d2, atol=1e-6)
