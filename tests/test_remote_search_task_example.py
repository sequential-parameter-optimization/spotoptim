# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for _remote_search_task function based on documentation examples.
These tests validate the parallel search functionality with dill serialization.
"""

import numpy as np
import dill
from spotoptim.SpotOptim import _remote_search_task
from spotoptim import SpotOptim


def test_remote_search_task_basic_example():
    """Test basic usage from documentation example."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=10,
        seed=0,
        verbose=False,  # Suppress output during testing
    )

    # Initialize the optimizer by manually setting data and fitting surrogate
    np.random.seed(0)
    opt.X_ = np.random.rand(10, 2) * 10 - 5  # Scale to bounds [-5, 5]
    opt.y_ = np.sum(opt.X_**2, axis=1)
    opt._fit_surrogate(opt.X_, opt.y_)

    pickled_optimizer = dill.dumps(opt)
    x_new = _remote_search_task(pickled_optimizer)

    # The function should return an infill point (ndarray), not an Exception
    assert isinstance(x_new, np.ndarray), f"Expected ndarray, got {type(x_new)}"
    assert x_new.shape == (1, 2), f"Expected shape (1, 2), got {x_new.shape}"

    # Check that the point is within bounds
    for i, (low, high) in enumerate([(-5, 5), (-5, 5)]):
        assert (
            low <= x_new[0, i] <= high
        ), f"Point {x_new[0, i]} out of bounds [{low}, {high}]"


def test_remote_search_task_1d_problem():
    """Test with 1D optimization problem."""
    opt = SpotOptim(
        fun=lambda X: X**2,
        bounds=[(-10, 10)],
        n_initial=3,
        max_iter=5,
        seed=42,
        verbose=False,
    )

    # Initialize the optimizer by manually setting data and fitting surrogate
    np.random.seed(42)
    opt.X_ = np.random.rand(8, 1) * 20 - 10  # Scale to bounds [-10, 10]
    opt.y_ = (opt.X_**2).ravel()
    opt._fit_surrogate(opt.X_, opt.y_)

    pickled_optimizer = dill.dumps(opt)
    x_new = _remote_search_task(pickled_optimizer)

    assert isinstance(x_new, np.ndarray)
    assert x_new.shape == (1, 1)
    assert -10 <= x_new[0, 0] <= 10


def test_remote_search_task_5d_problem():
    """Test with 5D optimization problem."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-3, 3)] * 5,
        n_initial=10,
        max_iter=15,
        seed=123,
        verbose=False,
    )

    # Initialize the optimizer by manually setting data and fitting surrogate
    np.random.seed(123)
    opt.X_ = np.random.rand(12, 5) * 6 - 3  # Scale to bounds [-3, 3]
    opt.y_ = np.sum(opt.X_**2, axis=1)
    opt._fit_surrogate(opt.X_, opt.y_)

    pickled_optimizer = dill.dumps(opt)
    x_new = _remote_search_task(pickled_optimizer)

    assert isinstance(x_new, np.ndarray)
    assert x_new.shape == (1, 5)

    # Check all dimensions are within bounds
    for i in range(5):
        assert -3 <= x_new[0, i] <= 3


def test_remote_search_task_custom_objective():
    """Test with custom objective function (Rosenbrock)."""

    def rosenbrock(X):
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2"""
        x = X[:, 0]
        y = X[:, 1]
        return (1 - x) ** 2 + 100 * (y - x**2) ** 2

    opt = SpotOptim(
        fun=rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        n_initial=6,
        max_iter=12,
        seed=99,
        verbose=False,
    )

    # Initialize the optimizer by manually setting data and fitting surrogate
    np.random.seed(99)
    opt.X_ = np.random.rand(10, 2) * 4 - 2  # Scale to bounds [-2, 2]
    opt.y_ = rosenbrock(opt.X_)
    opt._fit_surrogate(opt.X_, opt.y_)

    pickled_optimizer = dill.dumps(opt)
    x_new = _remote_search_task(pickled_optimizer)

    assert isinstance(x_new, np.ndarray)
    assert x_new.shape == (1, 2)
    assert -2 <= x_new[0, 0] <= 2
    assert -2 <= x_new[0, 1] <= 2


def test_remote_search_task_with_acquisition_ei():
    """Test with Expected Improvement acquisition function."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        acquisition="ei",  # Expected Improvement
        n_initial=5,
        max_iter=10,
        seed=777,
        verbose=False,
    )

    # Initialize the optimizer by manually setting data and fitting surrogate
    np.random.seed(777)
    opt.X_ = np.random.rand(10, 2) * 10 - 5  # Scale to bounds [-5, 5]
    opt.y_ = np.sum(opt.X_**2, axis=1)
    opt._fit_surrogate(opt.X_, opt.y_)

    pickled_optimizer = dill.dumps(opt)
    x_new = _remote_search_task(pickled_optimizer)

    assert isinstance(x_new, np.ndarray)
    assert x_new.shape == (1, 2)


def test_remote_search_task_different_seeds():
    """Test that different seeds produce different infill points."""
    results = []

    for seed in [0, 1, 2]:
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=5,
            max_iter=10,
            seed=seed,
            verbose=False,
        )

        # Initialize the optimizer by manually setting data and fitting surrogate
        np.random.seed(seed)
        opt.X_ = np.random.rand(10, 2) * 10 - 5  # Scale to bounds [-5, 5]
        opt.y_ = np.sum(opt.X_**2, axis=1)
        opt._fit_surrogate(opt.X_, opt.y_)

        pickled_optimizer = dill.dumps(opt)
        x_new = _remote_search_task(pickled_optimizer)
        results.append(x_new)

    # With different seeds, we should get different points
    # (though theoretically they could be the same by chance)
    assert not np.allclose(results[0], results[1]) or not np.allclose(
        results[1], results[2]
    )


def test_remote_search_task_error_handling():
    """Test error handling when optimizer fails."""

    # Create an optimizer with an invalid configuration that will fail
    class FailingOptimizer:
        def suggest_next_infill_point(self):
            raise ValueError("Intentional failure for testing")

    failing_opt = FailingOptimizer()
    pickled_optimizer = dill.dumps(failing_opt)

    result = _remote_search_task(pickled_optimizer)

    # Should return the Exception, not raise it
    assert isinstance(result, Exception)
    assert isinstance(result, ValueError)
    assert "Intentional failure" in str(result)


def test_remote_search_task_preserves_optimizer_state():
    """Test that the optimizer state is correctly preserved through pickling."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        n_initial=5,
        max_iter=10,
        seed=42,
        verbose=False,
    )

    # Initialize the optimizer by manually setting data and fitting surrogate
    np.random.seed(42)
    opt.X_ = np.random.rand(10, 2) * 10 - 5  # Scale to bounds [-5, 5]
    opt.y_ = np.sum(opt.X_**2, axis=1)
    opt._fit_surrogate(opt.X_, opt.y_)

    # Store original seed
    original_seed = opt.seed

    pickled_optimizer = dill.dumps(opt)
    x_new = _remote_search_task(pickled_optimizer)

    # Unpickle to verify state was preserved
    opt_unpickled = dill.loads(pickled_optimizer)
    assert opt_unpickled.seed == original_seed
    assert isinstance(x_new, np.ndarray)
