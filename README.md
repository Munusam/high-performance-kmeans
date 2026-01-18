# 📘 High-Performance K-Means Clustering (From Scratch)

A high-performance, from-scratch implementation of the K-Means clustering algorithm using Python and NumPy, developed as part of the Programming and Data Structures (Z5007) course.

This project emphasizes:

algorithmic efficiency

vectorized numerical computation

custom data structure design

fair benchmarking against standard libraries

No external machine learning libraries are used for the core algorithm.

---
# 🎓 Academic Context

Course: Z5007 – Programming and Data Structures

Programme: M.Tech Data Science & Artificial Intelligence

Institution: IIT Madras Zanzibar

Student: Munusamy M

Roll Number: ZDA25M011

---
# 🚀 Project Highlights

This project demonstrates how careful algorithm design and data structure integration can yield a competitive K-Means implementation without relying on black-box libraries.

# ✔ Key Features

1️⃣Vectorized Squared Euclidean Distance

  ⭐Difference-based formulation (baseline)

  ⭐Expansion-based formulation

  ∥𝑥−𝑐∥2=∥𝑥∥2+∥𝑐∥2−2𝑥⋅𝑐∥x−c∥2=∥x∥2+∥c∥2−2x⋅c

Final implementation uses the expansion approach for better performance

2️⃣Custom Data Structures (Built from Scratch)

⭐Binary Min-Heap (array-based)

⭐Hash Table with linear probing

3️⃣K-Means++ Initialization

⭐Improved centroid initialization

⭐Faster convergence

⭐Reduced sensitivity to poor local minima

4️⃣Full K-Means (Lloyd’s Algorithm)

⭐Vectorized assignment step

⭐Efficient centroid update

⭐Convergence detection using numerical tolerance

5️⃣Robust Edge Case Handling

⭐Empty clusters handled using cached distances and Min-Heap

⭐Deterministic behavior via fixed random seed

6️⃣Evaluation & Analysis

⭐WCSS (Within-Cluster Sum of Squares)

⭐Elbow Method for optimal K selection

⭐Fair benchmarking against scikit-learn (n_init = 1)

---
# 🧠 Design Philosophy

Performance first: avoid Python loops in critical paths

Transparency: every component is implemented and explainable

Modularity: clean separation between algorithms and data structures

Fair comparison: identical experimental conditions for benchmarking

---

# ⚙️ Requirements

Python 3.9+

numpy

pandas

matplotlib

scikit-learn
(used only for benchmarking, not for core implementation)

---
# Install dependencies
pip install -r requirements.txt

---
# ▶️ How to Run (Demonstrations)

1️⃣ Vectorized Distance Benchmark

python benchmarks/compare_distance.py

2️⃣ Min-Heap Demonstration

python benchmarks/demo_min_heap.py

3️⃣ K-Means++ Initialization

python benchmarks/demo_kmeans_plus_plus.py

4️⃣ Full K-Means Algorithm

python benchmarks/demo_kmeans.py

5️⃣ Elbow Method

python benchmarks/demo_elbow.py

6️⃣ Full Demo on Real Dataset

python benchmarks/final_real_data.py

---
# 📊 Sample Results (Real Dataset)

Dataset size: ~20,433 samples, 8 numerical features

WCSS error vs scikit-learn: ~1.4%

Runtime ratio (Custom / scikit-learn): ~0.27

Benchmarking performed under identical conditions with n_init = 1 for fairness.

---
# 🧪 Testing

Comprehensive unit tests are provided for:

distance computation

Min-Heap

Hash Table

K-Means++

K-Means algorithm

Run all tests:
pytest tests/

---
# 📌 Notes

The Min-Heap is not used for nearest-centroid search
(assignment is fully vectorized using argmin)

The Min-Heap is used in:

K-Means++ initialization

Empty-cluster recovery

Hash Table is used for efficient WCSS storage during Elbow Method

---
# 👤 Author

Munusamy M
M.Tech Data Science & Artificial Intelligence
IIT Madras Zanzibar

---
# ✅ Final Remark

This project demonstrates that efficient clustering algorithms can be implemented from scratch using sound programming practices, appropriate data structures, and numerical optimization techniques—achieving performance comparable to established libraries.


