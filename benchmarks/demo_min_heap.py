# benchmarks/demo_min_heap.py

from src.kmeans.min_heap import MinHeap

heap = MinHeap()

distances = [
    (4.5, "point_1"),
    (2.1, "point_2"),
    (7.8, "point_3"),
    (1.2, "point_4"),
]

print("Inserting distances:")
for d in distances:
    print("push:", d)
    heap.push(d)

print("\nExtracting in priority order:")
while not heap.is_empty():
    print(heap.pop())
