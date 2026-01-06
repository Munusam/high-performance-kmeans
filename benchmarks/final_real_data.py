# benchmarks/final_demo_real_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from src.kmeans.kmeans import KMeans
from src.kmeans.utils import elbow_method
from sklearn.cluster import KMeans as SklearnKMeans
#
# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv("E:/high-performance-kmeans/data/housing.csv")

# Drop non-numeric / target column
df = df.drop(columns=["median_house_value", "ocean_proximity"], errors="ignore")

# Handle missing values
df = df.dropna()

# Convert to NumPy
X = df.values.astype(float)

# Standardization (VERY IMPORTANT)
X = (X - X.mean(axis=0)) / X.std(axis=0)

print("Dataset shape:", X.shape)

# -----------------------------
# 2. Run Custom K-Means
# -----------------------------
K = 5
start = time.perf_counter()

km = KMeans(n_clusters=K, random_state=42)
labels = km.fit_predict(X)

custom_time = time.perf_counter() - start

print("\nCustom K-Means Results")
print("Centroids shape:", km.centroids_.shape)
print("WCSS:", km.inertia_)
print("Runtime:", custom_time)

# -----------------------------
# 3. Elbow Method
# -----------------------------
print("\nRunning Elbow Method...")
wcss_table = elbow_method(X, k_min=1, k_max=10)

Ks = list(range(1, 11))
WCSS = [wcss_table.get(k) for k in Ks]

plt.figure()
plt.plot(Ks, WCSS, marker="o")
plt.xlabel("K")
plt.ylabel("WCSS")
plt.title("Elbow Method (Custom K-Means)")
plt.show()

# -----------------------------
# 4. Compare with scikit-learn
# -----------------------------
start = time.perf_counter()

sk = SklearnKMeans(
    n_clusters=K,
    init="k-means++",
    n_init=1,
    random_state=42,
)
sk.fit(X)

sk_time = time.perf_counter() - start

error_pct = abs(km.inertia_ - sk.inertia_) / sk.inertia_ * 100

print("\nScikit-learn Comparison")
print("Sklearn WCSS:", sk.inertia_)
print("Sklearn Runtime:", sk_time)
print("WCSS Error (%):", error_pct)
print("Runtime Ratio (Custom / Sklearn):", custom_time / sk_time)
