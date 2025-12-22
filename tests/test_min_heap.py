# tests/test_min_heap.py

import pytest
from src.kmeans.min_heap import MinHeap


def test_heap_push_pop():
    heap = MinHeap()

    items = [(5, "a"), (3, "b"), (8, "c"), (1, "d")]
    for item in items:
        heap.push(item)

    result = []
    while not heap.is_empty():
        result.append(heap.pop())

    keys = [item[0] for item in result]
    assert keys == sorted(keys)


def test_heap_peek():
    heap = MinHeap()
    heap.push((10, "x"))
    heap.push((2, "y"))

    assert heap.peek() == (2, "y")


def test_pop_empty():
    heap = MinHeap()
    with pytest.raises(IndexError):
        heap.pop()
