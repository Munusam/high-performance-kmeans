# tests/test_elbow.py

import numpy as np
from src.kmeans.utils import elbow_method


def test_elbow_method():
    np.random.seed(0)
    X = np.random.rand(100, 3)

    table = elbow_method(X, k_min=1, k_max=5, random_state=0)

    for K in range(1, 6):
        assert K in table
        assert table.get(K) >= 0
