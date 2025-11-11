"""Tests for termination criteria in SpotOptim.

This module tests the termination behavior of the SpotOptim optimizer,
including max_iter (function evaluation budget) and max_time (runtime limit).
"""

import pytest
import numpy as np
import time
from scipy.optimize import OptimizeResult
from spotoptim.SpotOptim import SpotOptim


def simple_sphere(X):
    """Simple sphere function for testing."""
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


class TestMaxIterTermination:
    """Test suite for max_iter termination criterion."""

    def test_max_iter_includes_initial_design(self):
        """Test that max_iter includes initial design evaluations."""
        n_initial = 10
        max_iter = 30
        
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=max_iter,
            n_initial=n_initial,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Total evaluations should equal max_iter
        assert result.nfev == max_iter, f"Expected {max_iter} evaluations, got {result.nfev}"
        
        # Number of sequential iterations should be max_iter - n_initial
        expected_iterations = max_iter - n_initial
        assert result.nit == expected_iterations, \
            f"Expected {expected_iterations} iterations, got {result.nit}"
        
        # Check termination message
        assert "maximum evaluations" in result.message.lower()

    def test_max_iter_exactly_n_initial(self):
        """Test when max_iter equals n_initial (no sequential iterations)."""
        n_initial = 10
        
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=n_initial,
            n_initial=n_initial,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Should only perform initial design
        assert result.nfev == n_initial
        assert result.nit == 0  # No sequential iterations
        
    def test_max_iter_less_than_n_initial(self):
        """Test when max_iter is less than n_initial - should raise ValueError."""
        n_initial = 20
        max_iter = 15
        
        with pytest.raises(ValueError, match="max_iter.*must be >= n_initial"):
            optimizer = SpotOptim(
                fun=simple_sphere,
                bounds=[(-5, 5), (-5, 5)],
                max_iter=max_iter,
                n_initial=n_initial,
                seed=42,
                verbose=False,
            )

    def test_max_iter_with_custom_initial_design(self):
        """Test max_iter with user-provided initial design."""
        X0 = np.array([
            [1.0, 1.0],
            [-1.0, -1.0],
            [2.0, -2.0],
            [-2.0, 2.0],
            [0.5, 0.5]
        ])
        n_initial = X0.shape[0]
        max_iter = 15
        
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=max_iter,
            n_initial=n_initial,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize(X0=X0)
        
        # Total evaluations should equal max_iter
        assert result.nfev == max_iter
        assert result.nit == max_iter - n_initial


class TestMaxTimeTermination:
    """Test suite for max_time termination criterion."""

    def test_max_time_terminates_early(self):
        """Test that optimization stops when max_time is exceeded."""
        # Create a slow objective function
        def slow_sphere(X):
            time.sleep(0.1)  # 100ms per evaluation
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)
        
        max_time = 0.5 / 60  # 0.5 seconds = 0.00833 minutes
        
        optimizer = SpotOptim(
            fun=slow_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=100,  # Would take ~10 seconds without time limit
            n_initial=3,
            max_time=max_time,
            seed=42,
            verbose=False,
        )
        
        start_time = time.time()
        result = optimizer.optimize()
        elapsed_time = time.time() - start_time
        
        # Should terminate before completing all iterations
        assert result.nfev < 100, f"Expected early termination, got {result.nfev} evaluations"
        
        # Should stop approximately at max_time (with some tolerance for overhead)
        assert elapsed_time < max_time * 60 + 1.0, \
            f"Runtime {elapsed_time:.2f}s exceeded time limit by >1s"
        
        # Check termination message
        assert "time limit" in result.message.lower()

    def test_max_time_infinite_by_default(self):
        """Test that default max_time allows optimization to complete."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            seed=42,
            verbose=False,
        )
        
        # max_time should be np.inf by default
        assert optimizer.max_time == np.inf
        
        result = optimizer.optimize()
        
        # Should complete all iterations
        assert result.nfev == 15
        assert "maximum evaluations" in result.message.lower()

    def test_max_time_large_value_equivalent_to_infinite(self):
        """Test that very large max_time effectively disables time limit."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            max_time=1e6,  # 1 million minutes
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Should complete all iterations
        assert result.nfev == 15
        assert "maximum evaluations" in result.message.lower()


class TestCombinedTermination:
    """Test suite for combined max_iter and max_time termination."""

    def test_max_iter_reached_before_max_time(self):
        """Test termination when max_iter is reached first."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            max_time=10.0,  # 10 minutes - won't be reached
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Should terminate due to max_iter
        assert result.nfev == 15
        assert "maximum evaluations" in result.message.lower()

    def test_max_time_reached_before_max_iter(self):
        """Test termination when max_time is reached first."""
        def slow_sphere(X):
            time.sleep(0.15)  # 150ms per evaluation
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)
        
        max_time = 0.6 / 60  # 0.6 seconds in minutes
        
        optimizer = SpotOptim(
            fun=slow_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=100,  # Would take ~15 seconds
            n_initial=2,
            max_time=max_time,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Should terminate due to max_time before reaching max_iter
        assert result.nfev < 100, f"Expected early termination, got {result.nfev} evaluations"
        assert "time limit" in result.message.lower()


class TestBackwardCompatibility:
    """Test that changes maintain backward compatibility."""

    def test_old_behavior_without_max_time(self):
        """Test that omitting max_time gives expected behavior."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Should complete all iterations as before
        assert result.nfev == 20
        assert result.nit == 10
        assert result.success

    def test_result_structure_unchanged(self):
        """Test that OptimizeResult structure remains unchanged."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Check all expected attributes exist
        assert hasattr(result, 'x')
        assert hasattr(result, 'fun')
        assert hasattr(result, 'nfev')
        assert hasattr(result, 'nit')
        assert hasattr(result, 'success')
        assert hasattr(result, 'message')
        assert hasattr(result, 'X')
        assert hasattr(result, 'y')
        
        # Check types
        assert isinstance(result, OptimizeResult)
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, (int, float))
        assert isinstance(result.nfev, int)
        assert isinstance(result.nit, int)
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.X, np.ndarray)
        assert isinstance(result.y, np.ndarray)


class TestDimensionReductionWithTermination:
    """Test termination criteria work correctly with dimension reduction."""

    def test_max_iter_with_dimension_reduction(self):
        """Test that max_iter works correctly with fixed dimensions."""
        optimizer = SpotOptim(
            fun=simple_sphere,
            bounds=[(-5, 5), (2, 2), (-5, 5)],  # Middle dimension is fixed
            max_iter=20,
            n_initial=10,
            seed=42,
            verbose=False,
        )
        
        result = optimizer.optimize()
        
        # Should have dimension reduction active
        assert optimizer.red_dim
        
        # Should complete all iterations
        assert result.nfev == 20
        assert result.nit == 10
        
        # Result should be in full dimensions
        assert result.x.shape[0] == 3
        assert result.X.shape[1] == 3
        
        # Fixed dimension should have correct value
        assert result.x[1] == 2.0
        np.testing.assert_array_equal(result.X[:, 1], np.full(20, 2.0))
