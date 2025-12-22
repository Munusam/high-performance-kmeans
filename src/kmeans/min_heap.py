# src/kmeans/min_heap.py

class MinHeap:
    """
    Custom binary Min-Heap implementation.
    Stores (key, value) pairs, ordered by key.
    """

    def __init__(self):
        self.heap = []

    def __len__(self):
        return len(self.heap)

    def is_empty(self):
        return len(self.heap) == 0

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek from empty heap")
        return self.heap[0]

    def push(self, item):
        """
        Insert a new item into the heap.
        item: tuple (key, value)
        """
        self.heap.append(item)
        self._heapify_up(len(self.heap) - 1)

    def pop(self):
        """
        Remove and return the smallest item.
        """
        if self.is_empty():
            raise IndexError("Pop from empty heap")

        root = self.heap[0]
        last_item = self.heap.pop()

        if not self.is_empty():
            self.heap[0] = last_item
            self._heapify_down(0)

        return root

    # ---------------- Internal Helpers ---------------- #

    def _heapify_up(self, index):
        parent = (index - 1) // 2

        while index > 0 and self.heap[index][0] < self.heap[parent][0]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def _heapify_down(self, index):
        size = len(self.heap)

        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index

            if left < size and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left

            if right < size and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right

            if smallest == index:
                break

            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest
