import pytest
import numpy as np
from unittest.mock import patch
from spotoptim.function.mohyperlight import MoHyperLight


@pytest.fixture
def mock_fun_control():
    return {"show_config": False, "verbosity": 0, "weights": 1.0}


@patch("spotpython.fun.mohyperlight.MoHyperLight.fun")
def test_mohyperlight_wrapper(mock_base_fun, mock_fun_control):
    # Setup mock return value
    # Expecting (n, 2) return from base fun
    mock_base_fun.return_value = np.array([[0.5, 10], [0.4, 20]])

    # Initialize wrapper
    wrapper = MoHyperLight(fun_control=mock_fun_control, seed=42)

    # Test fun call without explicit fun_control
    X = np.array([[1, 2], [3, 4]])
    result = wrapper.fun(X)

    # Verify base fun was called with correct arguments
    mock_base_fun.assert_called_once()
    args, _ = mock_base_fun.call_args
    assert np.array_equal(args[0], X)
    assert args[1] == mock_fun_control

    # Verify result
    assert np.array_equal(result, np.array([[0.5, 10], [0.4, 20]]))


@patch("spotpython.fun.mohyperlight.MoHyperLight.fun")
def test_mohyperlight_wrapper_explicit_control(mock_base_fun, mock_fun_control):
    mock_base_fun.return_value = np.array([[0.5, 10]])

    wrapper = MoHyperLight(fun_control=mock_fun_control)

    other_control = {"show_config": True}
    X = np.array([[1, 2]])
    wrapper.fun(X, fun_control=other_control)

    # Verify base fun called with EXPLICIT control
    args, _ = mock_base_fun.call_args
    assert args[1] == other_control
