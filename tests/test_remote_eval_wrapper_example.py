# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import numpy as np
import dill

from spotoptim.SpotOptim import _remote_eval_wrapper


class DummyOptimizer:
    """Simple optimizer for testing purposes."""

    def _evaluate_function(self, X):
        """Evaluate sum of squares."""
        return np.sum(X**2, axis=1)


class FailingOptimizer:
    """Optimizer that always raises an error."""

    def _evaluate_function(self, X):
        raise ValueError("Intentional evaluation error")


def test_remote_eval_wrapper_example():
    """Test _remote_eval_wrapper with basic example from documentation."""
    optimizer = DummyOptimizer()
    x = np.array([1.0, 2.0])

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    assert np.allclose(x_eval, x)
    assert np.isclose(y_eval, 5.0)  # 1^2 + 2^2 = 5


def test_remote_eval_wrapper_single_dimension():
    """Test _remote_eval_wrapper with 1D input."""
    optimizer = DummyOptimizer()
    x = np.array([3.0])

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    assert np.allclose(x_eval, x)
    assert np.isclose(y_eval, 9.0)  # 3^2 = 9


def test_remote_eval_wrapper_high_dimensional():
    """Test _remote_eval_wrapper with high-dimensional input."""
    optimizer = DummyOptimizer()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    assert np.allclose(x_eval, x)
    # 1 + 4 + 9 + 16 + 25 = 55
    assert np.isclose(y_eval, 55.0)


def test_remote_eval_wrapper_zero_point():
    """Test _remote_eval_wrapper at the origin."""
    optimizer = DummyOptimizer()
    x = np.array([0.0, 0.0, 0.0])

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    assert np.allclose(x_eval, x)
    assert np.isclose(y_eval, 0.0)


def test_remote_eval_wrapper_negative_values():
    """Test _remote_eval_wrapper with negative values."""
    optimizer = DummyOptimizer()
    x = np.array([-2.0, -3.0])

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    assert np.allclose(x_eval, x)
    # (-2)^2 + (-3)^2 = 4 + 9 = 13
    assert np.isclose(y_eval, 13.0)


def test_remote_eval_wrapper_error_handling():
    """Test _remote_eval_wrapper handles evaluation errors correctly."""
    optimizer = FailingOptimizer()
    x = np.array([1.0, 2.0])

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    # When an exception occurs, x_eval should be None
    assert x_eval is None
    # y_eval should be the exception
    assert isinstance(y_eval, Exception)
    assert isinstance(y_eval, ValueError)
    assert "Intentional evaluation error" in str(y_eval)


def test_remote_eval_wrapper_with_custom_function():
    """Test _remote_eval_wrapper with a different objective function."""

    class RosenbrockOptimizer:
        """Optimizer using Rosenbrock function."""

        def _evaluate_function(self, X):
            """Rosenbrock function: (1-x)^2 + 100*(y-x^2)^2"""
            x = X[:, 0]
            y = X[:, 1]
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    optimizer = RosenbrockOptimizer()
    x = np.array([0.0, 0.0])

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    assert np.allclose(x_eval, x)
    # At (0,0): (1-0)^2 + 100*(0-0)^2 = 1
    assert np.isclose(y_eval, 1.0)


def test_remote_eval_wrapper_preserves_input():
    """Test that _remote_eval_wrapper preserves the input point."""
    optimizer = DummyOptimizer()
    original_x = np.array([1.5, 2.5, 3.5])
    x = original_x.copy()

    pickled_args = dill.dumps((optimizer, x))
    x_eval, y_eval = _remote_eval_wrapper(pickled_args)

    # Verify the returned x matches the input
    assert np.array_equal(x_eval, original_x)
    # Verify original array wasn't modified
    assert np.array_equal(x, original_x)
