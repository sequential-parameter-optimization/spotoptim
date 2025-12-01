"""Tests for max_iter and n_initial validation in SpotOptim."""

import pytest
import numpy as np
from spotoptim.SpotOptim import SpotOptim


def simple_sphere(X):
    """Simple sphere function for testing."""
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


class TestMaxIterValidation:
    """Test suite for max_iter and n_initial validation."""

    def test_max_iter_less_than_n_initial_raises_error(self):
        """Test that max_iter < n_initial raises ValueError."""
        with pytest.raises(ValueError, match="max_iter.*must be >= n_initial"):
            SpotOptim(
                fun=simple_sphere,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=5,
                n_initial=10,  # n_initial > max_iter
                seed=42,
            )

    def test_max_iter_equals_n_initial_works(self):
        """Test that max_iter == n_initial works correctly (returns best from initial design)."""
        n_initial = 10

        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=n_initial,  # Equal to n_initial
            n_initial=n_initial,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Should complete successfully
        assert result.success

        # Should only perform initial design (no sequential iterations)
        assert result.nfev == n_initial
        assert result.nit == 0

        # Should have valid result
        assert result.x is not None
        assert result.fun is not None
        assert isinstance(result.fun, (int, float))
        assert result.fun >= 0  # Sphere function is always >= 0

        # Check that result came from initial design
        assert result.X.shape[0] == n_initial
        assert result.y.shape[0] == n_initial

    def test_max_iter_greater_than_n_initial_works(self):
        """Test normal case where max_iter > n_initial."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Should complete successfully
        assert result.success

        # Should perform sequential iterations
        assert result.nfev == 20
        assert result.nit == 10  # 20 - 10 = 10 sequential iterations

    def test_error_message_includes_values(self):
        """Test that error message includes actual values."""
        max_iter = 3
        n_initial = 7

        with pytest.raises(ValueError) as exc_info:
            SpotOptim(
                fun=simple_sphere,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=max_iter,
                n_initial=n_initial,
                seed=42,
            )

        error_message = str(exc_info.value)
        assert str(max_iter) in error_message
        assert str(n_initial) in error_message
        assert "total function evaluation budget" in error_message

    def test_max_iter_one_n_initial_one_works(self):
        """Test edge case where both are 1."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=1,
            n_initial=1,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert result.success
        assert result.nfev == 1
        assert result.nit == 0
        assert result.x.shape[0] == 2  # 2D problem
        assert result.fun >= 0

    def test_custom_initial_design_with_equal_max_iter(self):
        """Test that custom initial design works when max_iter == n_initial."""
        X0 = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -2.0], [-2.0, 2.0], [0.5, 0.5]])
        n_points = X0.shape[0]

        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=n_points,
            n_initial=n_points,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize(X0=X0)

        assert result.success
        assert result.nfev == n_points
        assert result.nit == 0

        # Best should be the point closest to origin
        assert result.fun == 0.5  # [0.5, 0.5] -> 0.5^2 + 0.5^2 = 0.5
