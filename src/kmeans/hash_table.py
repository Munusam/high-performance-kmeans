# src/kmeans/hash_table.py

class HashTable:
    """
    Custom Hash Table using Open Addressing with Linear Probing.
    """

    def __init__(self, capacity=101):
        self.capacity = capacity
        self.size = 0
        self.keys = [None] * capacity
        self.values = [None] * capacity
        self.DELETED = object()

    def _hash(self, key):
        return hash(key) % self.capacity

    def _probe(self, index):
        return (index + 1) % self.capacity

    def put(self, key, value):
        if self.size >= self.capacity * 0.7:
            raise RuntimeError("HashTable is full or needs resizing")

        index = self._hash(key)

        while self.keys[index] is not None and self.keys[index] is not self.DELETED:
            if self.keys[index] == key:
                self.values[index] = value
                return
            index = self._probe(index)

        self.keys[index] = key
        self.values[index] = value
        self.size += 1

    def get(self, key):
        index = self._hash(key)
        start = index

        while self.keys[index] is not None:
            if self.keys[index] == key:
                return self.values[index]
            index = self._probe(index)
            if index == start:
                break

        raise KeyError(key)

    def delete(self, key):
        index = self._hash(key)
        start = index

        while self.keys[index] is not None:
            if self.keys[index] == key:
                self.keys[index] = self.DELETED
                self.values[index] = None
                self.size -= 1
                return
            index = self._probe(index)
            if index == start:
                break

        raise KeyError(key)

    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def __len__(self):
        return self.size
