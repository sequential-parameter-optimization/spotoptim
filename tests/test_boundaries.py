import numpy as np
import pytest
from spotoptim.utils.boundaries import get_boundaries

def test_get_boundaries_basic():
    data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    min_vals, max_vals = get_boundaries(data)
    assert np.array_equal(min_vals, [1, 2, 3])
    assert np.array_equal(max_vals, [7, 8, 9])

def test_get_boundaries_single_row():
    data = np.array([[10, 20, 30]])
    min_vals, max_vals = get_boundaries(data)
    assert np.array_equal(min_vals, [10, 20, 30])
    assert np.array_equal(max_vals, [10, 20, 30])

def test_get_boundaries_negative_values():
    data = np.array([[-1, -2, -3], [0, 0, 0], [1, 2, 3]])
    min_vals, max_vals = get_boundaries(data)
    assert np.array_equal(min_vals, [-1, -2, -3])
    assert np.array_equal(max_vals, [1, 2, 3])

def test_get_boundaries_empty_array():
    data = np.array([]).reshape(1, 0)
    with pytest.raises(ValueError):
        get_boundaries(data)

def test_get_boundaries_column_vector():
    data = np.array([[5], [10], [3]])
    min_vals, max_vals = get_boundaries(data)
    assert np.array_equal(min_vals, [3])
    assert np.array_equal(max_vals, [10])
