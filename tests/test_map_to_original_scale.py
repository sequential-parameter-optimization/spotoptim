import numpy as np
import pytest
from spotoptim.utils.boundaries import map_to_original_scale


def test_map_to_original_scale_basic():
    X_search = np.array([[0.0, 0.5, 1.0], [1.0, 0.0, 0.5]])
    x_min = np.array([10, 20, 30])
    x_max = np.array([20, 40, 60])
    expected = np.array([[10, 30, 60], [20, 20, 45]])
    result = map_to_original_scale(X_search, x_min, x_max)
    assert np.allclose(result, expected)


def test_map_to_original_scale_single_value():
    X_search = np.array([[0.5]])
    x_min = np.array([0])
    x_max = np.array([10])
    expected = np.array([[5]])
    result = map_to_original_scale(X_search, x_min, x_max)
    assert np.allclose(result, expected)


def test_map_to_original_scale_vector():
    X_search = np.array([[0.0], [1.0], [0.5]])
    x_min = np.array([1])
    x_max = np.array([3])
    expected = np.array([[1], [3], [2]])
    result = map_to_original_scale(X_search, x_min, x_max)
    assert np.allclose(result, expected)


def test_map_to_original_scale_shape_mismatch():
    X_search = np.array([[0.0, 0.5]])
    x_min = np.array([0])
    x_max = np.array([10])
    with pytest.raises(IndexError):
        map_to_original_scale(X_search, x_min, x_max)
