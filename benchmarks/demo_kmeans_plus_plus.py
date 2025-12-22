# benchmarks/demo_kmeans_plus_plus.py

import numpy as np
from src.kmeans.kmeans_plus_plus import kmeans_plus_plus

np.random.seed(0)
X = np.random.rand(20, 2)

C = kmeans_plus_plus(X, K=3, random_state=42)

print("Initialized centroids:")
print(C)
