# High Performance K-Means Clustering (From Scratch)

This repository contains a high-performance implementation of the K-Means clustering
algorithm built entirely from scratch using Python and NumPy, without relying on
external machine learning libraries for the core algorithm.

This project is part of the course **Z5007: Programming and Data Structures**
(M.Tech Data Science & Artificial Intelligence, IIT Madras Zanzibar).

---

## 📌 Features Implemented

- Vectorized squared Euclidean distance computation
  - Difference-based formulation
  - Expansion-based formulation (final choice)
- Custom data structures
  - Binary Min-Heap (array-based)
  - Hash Table with linear probing
- K-Means++ initialization (from scratch)
- Full K-Means clustering (Lloyd’s Algorithm)
- Robust empty-cluster handling using cached distances + Min-Heap
- Elbow Method for optimal K selection
- WCSS (Inertia) computation
- Benchmarking against scikit-learn



## ⚙️ Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn (used **only** for benchmarking)

Install dependencies:
```bash
pip install -r requirements.txt

▶️ How to Run (Milestone-2 Demo)
1. Vectorized Distance Benchmark
python benchmarks/compare_distance.py

2. Min-Heap Demo
python benchmarks/demo_min_heap.py

3. K-Means++ Demo
python benchmarks/demo_kmeans_plus_plus.py

4. Full K-Means Demo
python benchmarks/demo_kmeans.py

5. Elbow Method
python benchmarks/demo_elbow.py

6. Full Demo on Real Dataset
python benchmarks/final_real_data.py

---
📊 Sample Results (Real Dataset)

Dataset size: ~20,000 samples

WCSS error vs scikit-learn: ~1.4%

Runtime ratio (Custom / sklearn): ~0.27
---
🧪 Testing

Run all unit tests:

pytest tests/
---

👤 Author

Munusamy M
M.Tech Data Science & Artificial Intelligence
IIT Madras Zanzibar



