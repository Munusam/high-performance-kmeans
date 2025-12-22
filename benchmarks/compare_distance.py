# benchmarks/compare_distance.py

import numpy as np
import timeit
from src.kmeans.distance import (
    squared_euclidean_difference,
    squared_euclidean_expansion
)

# Dataset size similar to California Housing
N, F, K = 20000, 9, 10
np.random.seed(0)

X = np.random.rand(N, F)
C = np.random.rand(K, F)

# Warm-up
squared_euclidean_difference(X, C)
squared_euclidean_expansion(X, C)

# Timing
diff_time = timeit.timeit(
    lambda: squared_euclidean_difference(X, C),
    number=5
)

exp_time = timeit.timeit(
    lambda: squared_euclidean_expansion(X, C),
    number=5
)

print("Difference-based time:", diff_time)
print("Expansion-based time:", exp_time)
print("Speedup:", diff_time / exp_time)
