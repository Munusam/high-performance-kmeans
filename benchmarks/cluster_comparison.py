import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.decomposition import PCA

from src.kmeans.kmeans import KMeans

# -----------------------------
# Load and preprocess dataset
# -----------------------------
df = pd.read_csv("E:/high-performance-kmeans/data/housing.csv")
df = df.drop(columns=["median_house_value", "ocean_proximity"], errors="ignore")
df = df.dropna()

X = df.values.astype(float)
X = (X - X.mean(axis=0)) / X.std(axis=0)

K = 5

# -----------------------------
# Run Custom K-Means
# -----------------------------
custom_km = KMeans(n_clusters=K, random_state=42)
custom_labels = custom_km.fit_predict(X)
custom_centroids = custom_km.centroids_

# -----------------------------
# Run scikit-learn K-Means
# -----------------------------
sk_km = SklearnKMeans(
    n_clusters=K,
    init="k-means++",
    n_init=1,
    random_state=42
)
sk_labels = sk_km.fit_predict(X)
sk_centroids = sk_km.cluster_centers_

# -----------------------------
# PCA for visualization (2D)
# -----------------------------
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# -----------------------------
# Compute per-point distance difference
# -----------------------------
def dist_to_centroid(X, labels, centroids):
    return np.linalg.norm(X - centroids[labels], axis=1)

custom_dist = dist_to_centroid(X, custom_labels, custom_centroids)
sk_dist = dist_to_centroid(X, sk_labels, sk_centroids)

dist_diff = custom_dist - sk_dist

# -----------------------------
# Plot 1: Custom K-Means clusters
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1],
            c=custom_labels, cmap="tab10", s=8)
plt.title("Custom K-Means Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: scikit-learn clusters
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1],
            c=sk_labels, cmap="tab10", s=8)
plt.title("scikit-learn K-Means Clustering (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 3: Error / Difference Map
# -----------------------------
plt.figure(figsize=(7, 6))
plt.scatter(X_2d[:, 0], X_2d[:, 1],
            c=dist_diff, cmap="coolwarm", s=8)
plt.colorbar(label="Distance Difference (Custom - Sklearn)")
plt.title("Cluster Assignment Difference Map")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

print("Mean distance difference:", np.mean(dist_diff))
print("Std of distance difference:", np.std(dist_diff))
