# benchmarks/compare_sklearn.py

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as SklearnKMeans

from src.kmeans.kmeans import KMeans


def benchmark(X, K, random_state=42):
    # ---------- Custom K-Means ----------
    start = time.perf_counter()
    km_custom = KMeans(
        n_clusters=K,
        random_state=random_state,
        tol=1e-6,
        max_iter=300,
    )
    km_custom.fit(X)
    custom_time = time.perf_counter() - start
    custom_wcss = km_custom.inertia_

    # ---------- Scikit-learn K-Means ----------
    start = time.perf_counter()
    km_sklearn = SklearnKMeans(
        n_clusters=K,
        init="k-means++",
        n_init=1,
        random_state=random_state,
        tol=1e-6,
        max_iter=300,
    )
    km_sklearn.fit(X)
    sklearn_time = time.perf_counter() - start
    sklearn_wcss = km_sklearn.inertia_

    # ---------- Error ----------
    wcss_error_pct = abs(custom_wcss - sklearn_wcss) / sklearn_wcss * 100

    return custom_time, sklearn_time, custom_wcss, sklearn_wcss, wcss_error_pct


if __name__ == "__main__":
    np.random.seed(0)

    # Dataset size similar to proposal
    N, F = 20000, 9
    X = np.random.rand(N, F)

    K = 5

    (
        custom_time,
        sklearn_time,
        custom_wcss,
        sklearn_wcss,
        error_pct,
    ) = benchmark(X, K)

    print("==== Benchmark Results ====")
    print(f"Custom K-Means time    : {custom_time:.4f} s")
    print(f"Sklearn K-Means time   : {sklearn_time:.4f} s")
    print(f"Custom WCSS            : {custom_wcss:.4f}")
    print(f"Sklearn WCSS           : {sklearn_wcss:.4f}")
    print(f"WCSS error (%)         : {error_pct:.2f}%")

    print("\nPerformance ratio (Custom / Sklearn):",
          custom_time / sklearn_time)
