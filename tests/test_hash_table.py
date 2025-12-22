# tests/test_hash_table.py

import pytest
from src.kmeans.hash_table import HashTable


def test_put_get():
    ht = HashTable()

    ht.put(1, "a")
    ht.put(2, "b")
    ht.put(3, "c")

    assert ht.get(1) == "a"
    assert ht.get(2) == "b"
    assert ht.get(3) == "c"


def test_update_value():
    ht = HashTable()
    ht.put(1, "a")
    ht.put(1, "updated")

    assert ht.get(1) == "updated"


def test_delete():
    ht = HashTable()
    ht.put(10, "x")
    ht.delete(10)

    with pytest.raises(KeyError):
        ht.get(10)


def test_contains():
    ht = HashTable()
    ht.put("k", 123)

    assert "k" in ht
    assert "missing" not in ht
