"""
Tests for acquisition failure handling functionality.

This test suite validates the acquisition_failure_strategy parameter and
the _handle_acquisition_failure() method, ensuring the "random" strategy
works correctly.
"""

import pytest
import numpy as np
from spotoptim.SpotOptim import SpotOptim


class TestAcquisitionFailureParameter:
    """Test suite for acquisition_failure_strategy parameter."""

    def test_default_strategy_is_random(self):
        """Test that default acquisition_failure_strategy is 'random'."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        assert optimizer.acquisition_failure_strategy == "random"

    def test_set_strategy_to_random(self):
        """Test explicitly setting acquisition_failure_strategy to 'random'."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            acquisition_failure_strategy="random",
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        assert optimizer.acquisition_failure_strategy == "random"

    def test_invalid_strategy_still_works(self):
        """Test that invalid strategy falls back gracefully (defaults to random in method)."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        # Note: No validation in __init__, so invalid values are stored
        # The _handle_acquisition_failure method treats anything != "mm" as random
        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            acquisition_failure_strategy="invalid",
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        assert optimizer.acquisition_failure_strategy == "invalid"


class TestHandleAcquisitionFailureMethod:
    """Test suite for _handle_acquisition_failure() method."""

    def test_handle_acquisition_failure_random_returns_valid_point(self):
        """Test that random strategy returns a valid point within bounds."""

        def simple_func(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=[(-5, 5), (-5, 5)],
            acquisition_failure_strategy="random",
            max_iter=10,
            n_initial=5,
            seed=42,
        )

        # Run initial design to populate X_
        optimizer.optimize()

        # Test the method directly
        x_new = optimizer._handle_acquisition_failure()

        # Check shape
        assert x_new.shape == (2,), f"Expected shape (2,), got {x_new.shape}"

        # Check bounds
        assert -5 <= x_new[0] <= 5, f"x[0]={x_new[0]} outside bounds [-5, 5]"
        assert -5 <= x_new[1] <= 5, f"x[1]={x_new[1]} outside bounds [-5, 5]"


class TestAcquisitionFailureIntegration:
    """Integration tests for acquisition failure handling in full optimization."""

    def test_optimization_with_random_strategy(self):
        """Test full optimization run with random acquisition failure strategy."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        optimizer = SpotOptim(
            fun=sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=10,
            acquisition_failure_strategy="random",
            tolerance_x=0.2,  # Moderate tolerance to potentially trigger failures
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Check that optimization succeeded
        assert result.success is True
        assert result.nfev == 30
        assert result.fun < 1.0  # Should find good solution


class TestAcquisitionFailureWithVariableTypes:
    """Test acquisition failure handling with different variable types."""

    def test_acquisition_failure_with_integer_variables(self):
        """Test fallback strategies respect integer variable constraints."""

        def int_func(X):
            return np.sum(X**2, axis=1)

        # Test both strategies
        for strategy in ["random"]:
            optimizer = SpotOptim(
                fun=int_func,
                bounds=[(-5, 5), (-5, 5)],
                var_type=["int", "int"],
                acquisition_failure_strategy=strategy,
                max_iter=15,
                n_initial=5,
                seed=42,
                verbose=False,
            )

            result = optimizer.optimize()

            # Check that all points are integers
            for x in optimizer.X_:
                assert np.all(
                    x == np.round(x)
                ), f"Strategy '{strategy}': Expected integer values, got {x}"

            # Check final result
            assert result.success is True
            assert np.all(result.x == np.round(result.x))

    def test_acquisition_failure_with_factor_variables(self):
        """Test fallback strategies respect factor variable constraints."""

        def factor_func(X):
            return np.sum((X - 2) ** 2, axis=1)

        # Test both strategies
        for strategy in ["random"]:
            optimizer = SpotOptim(
                fun=factor_func,
                bounds=[(0, 4), (0, 4)],
                var_type=["factor", "factor"],
                acquisition_failure_strategy=strategy,
                max_iter=15,
                n_initial=5,
                seed=42,
                verbose=False,
            )

            result = optimizer.optimize()

            # Check that all points are integers (factors)
            for x in optimizer.X_:
                assert np.all(
                    x == np.round(x)
                ), f"Strategy '{strategy}': Expected factor (int) values, got {x}"

            # Check final result
            assert result.success is True

    def test_acquisition_failure_with_mixed_variables(self):
        """Test fallback strategies with mixed variable types."""

        def mixed_func(X):
            return np.sum(X**2, axis=1)

        # Test both strategies
        for strategy in ["random"]:
            optimizer = SpotOptim(
                fun=mixed_func,
                bounds=[(-5, 5), (-5, 5), (-5, 5)],
                var_type=["float", "int", "factor"],
                acquisition_failure_strategy=strategy,
                max_iter=20,
                n_initial=8,
                seed=42,
                verbose=False,
            )

            result = optimizer.optimize()

            # Check variable types in all evaluated points
            for x in optimizer.X_:
                # x[0] is float - can be any float
                # x[1] is int - must be integer
                assert x[1] == np.round(
                    x[1]
                ), f"Strategy '{strategy}': x[1] should be int, got {x[1]}"
                # x[2] is factor - must be integer
                assert x[2] == np.round(
                    x[2]
                ), f"Strategy '{strategy}': x[2] should be factor (int), got {x[2]}"

            assert result.success is True


class TestAcquisitionFailureEdgeCases:
    """Test edge cases and robustness of acquisition failure handling."""

    def test_fallback_with_one_dimensional_problem(self):
        """Test fallback strategies work with 1D problems."""

        def func_1d(X):
            return X**2

        for strategy in ["random"]:
            optimizer = SpotOptim(
                fun=func_1d,
                bounds=[(-5, 5)],
                acquisition_failure_strategy=strategy,
                max_iter=15,
                n_initial=5,
                seed=42,
                verbose=False,
            )

            result = optimizer.optimize()

            assert result.success is True
            assert result.x.shape == (1,)
            assert -5 <= result.x[0] <= 5

    def test_fallback_with_high_dimensional_problem(self):
        """Test fallback strategies work with high-dimensional problems."""

        def func_high_dim(X):
            return np.sum(X**2, axis=1)

        n_dim = 10
        bounds = [(-5, 5)] * n_dim

        for strategy in ["random"]:
            optimizer = SpotOptim(
                fun=func_high_dim,
                bounds=bounds,
                acquisition_failure_strategy=strategy,
                max_iter=30,
                n_initial=15,
                seed=42,
                verbose=False,
            )

            result = optimizer.optimize()

            assert result.success is True
            assert result.x.shape == (n_dim,)
            assert np.all(result.x >= -5)
            assert np.all(result.x <= 5)

    def test_fallback_with_asymmetric_bounds(self):
        """Test fallback strategies respect asymmetric bounds."""

        def func_asym(X):
            return np.sum(X**2, axis=1)

        for strategy in ["random"]:
            optimizer = SpotOptim(
                fun=func_asym,
                bounds=[(-10, 2), (0, 15), (-5, 5)],
                acquisition_failure_strategy=strategy,
                max_iter=20,
                n_initial=10,
                seed=42,
                verbose=False,
            )

            result = optimizer.optimize()

            # Check all points respect bounds
            for x in optimizer.X_:
                assert -10 <= x[0] <= 2, f"x[0]={x[0]} outside [-10, 2]"
                assert 0 <= x[1] <= 15, f"x[1]={x[1]} outside [0, 15]"
                assert -5 <= x[2] <= 5, f"x[2]={x[2]} outside [-5, 5]"

            assert result.success is True

    def test_reproducibility_with_seed(self):
        """Test that acquisition failure strategies are reproducible with seed."""

        def func(X):
            return np.sum(X**2, axis=1)

        # Run twice with same seed
        results = []
        for _ in range(2):
            optimizer = SpotOptim(
                fun=func,
                bounds=[(-5, 5), (-5, 5)],
                acquisition_failure_strategy="random",
                max_iter=20,
                n_initial=8,
                tolerance_x=0.5,  # Higher tolerance to trigger fallbacks
                seed=123,
                verbose=False,
            )
            result = optimizer.optimize()
            results.append(result)

        # Results should be identical
        np.testing.assert_array_almost_equal(
            results[0].x,
            results[1].x,
            decimal=10,
            err_msg="Results with same seed should be identical",
        )
        assert abs(results[0].fun - results[1].fun) < 1e-10
