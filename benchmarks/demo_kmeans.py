# benchmarks/demo_kmeans.py

import numpy as np
from src.kmeans.kmeans import KMeans

np.random.seed(0)
X = np.random.rand(200, 2)

km = KMeans(n_clusters=3, random_state=42)
labels = km.fit_predict(X)

print("Centroids:")
print(km.centroids_)
print("Inertia (WCSS):", km.inertia_)
