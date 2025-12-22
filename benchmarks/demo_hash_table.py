# benchmarks/demo_hash_table.py

from src.kmeans.hash_table import HashTable

ht = HashTable()

print("Storing WCSS values:")
ht.put(2, 150.4)
ht.put(3, 98.7)
ht.put(4, 72.1)

print("WCSS for K=2:", ht.get(2))
print("WCSS for K=3:", ht.get(3))
print("WCSS for K=4:", ht.get(4))
