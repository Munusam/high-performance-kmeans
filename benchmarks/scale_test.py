# benchmarks/scale_test.py

import numpy as np
import time
import matplotlib.pyplot as plt
from src.kmeans.kmeans import KMeans

sizes = [1000, 5000, 10000, 20000]
times = []

np.random.seed(0)

for N in sizes:
    X = np.random.rand(N, 9)

    km = KMeans(n_clusters=5, random_state=0)

    start = time.perf_counter()
    km.fit(X)
    elapsed = time.perf_counter() - start

    times.append(elapsed)
    print(f"N={N}, Time={elapsed:.4f}s")

plt.plot(sizes, times, marker="o")
plt.xlabel("Number of samples (N)")
plt.ylabel("Execution time (seconds)")
plt.title("Scaling Behavior of Custom K-Means")
plt.show()
