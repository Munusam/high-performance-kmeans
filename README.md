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

---
## 📁 Project Structure
high-performance-kmeans/
-│
-├── src/
-│ └── kmeans/
-│ ├── distance.py
-│ ├── min_heap.py
-│ ├── hash_table.py
-│ ├── kmeans_plus_plus.py
-│ ├── kmeans.py
-│ └── utils.py
-│
-├── benchmarks/
-│ ├── compare_distance.py
-│ ├── demo_min_heap.py
-│ ├── demo_kmeans_plus_plus.py
-│ ├── demo_kmeans.py
-│ ├── demo_elbow.py
-│ └── final_real_data.py
-│
-├── tests/
-│ ├── test_distance.py
-│ ├── test_min_heap.py
-│ ├── test_hash_table.py
-│ ├── test_kmeans_plus_plus.py
-│ └── test_kmeans.py
-│
-├── data/
-│ └── sample_data.csv (optional small sample)
-│
-├── README.md
-└── requirements.txt
----

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


