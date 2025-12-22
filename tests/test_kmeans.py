# tests/test_kmeans.py

import numpy as np
from src.kmeans.kmeans import KMeans


def test_kmeans_basic():
    np.random.seed(0)
    X = np.random.rand(100, 2)

    km = KMeans(n_clusters=3, random_state=0)
    labels = km.fit_predict(X)

    assert labels.shape == (100,)
    assert km.centroids_.shape == (3, 2)
    assert km.inertia_ >= 0


def test_kmeans_deterministic():
    X = np.random.rand(80, 4)

    km1 = KMeans(n_clusters=4, random_state=42)
    km2 = KMeans(n_clusters=4, random_state=42)

    km1.fit(X)
    km2.fit(X)

    assert np.allclose(km1.centroids_, km2.centroids_)
