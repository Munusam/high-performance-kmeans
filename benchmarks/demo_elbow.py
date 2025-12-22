# benchmarks/demo_elbow.py

import numpy as np
import matplotlib.pyplot as plt
from src.kmeans.utils import elbow_method

np.random.seed(0)
X = np.random.rand(500, 2)

wcss_table = elbow_method(X, k_min=1, k_max=10)

Ks = list(range(1, 11))
WCSS = [wcss_table.get(k) for k in Ks]

plt.plot(Ks, WCSS, marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method (Custom K-Means)")
plt.show()
