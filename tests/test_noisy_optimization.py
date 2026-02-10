# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for noisy function evaluation and statistics tracking in SpotOptim."""

import pytest
import numpy as np
from spotoptim.SpotOptim import SpotOptim


def noisy_sphere(X, sigma=0.1, seed=None):
    """Noisy sphere function for testing."""
    if seed is not None:
        np.random.seed(seed)
    X = np.atleast_2d(X)
    base_values = np.sum(X**2, axis=1)
    noise = np.random.normal(0, sigma, size=base_values.shape)
    return base_values + noise


class TestNoisyOptimization:
    """Test suite for noisy function evaluation."""

    def test_repeats_initial_parameter(self):
        """Test that repeats_initial parameter is properly initialized."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            repeats_initial=3,
        )
        assert opt.repeats_initial == 3
        assert opt.repeats_initial == 3

    def test_repeats_surrogate_parameter(self):
        """Test that repeats_surrogate parameter is properly initialized."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            repeats_surrogate=2,
        )
        assert opt.repeats_surrogate == 2
        assert opt.repeats_surrogate == 2

    def test_noise_flag_activation(self):
        """Test that noise flag is set correctly."""
        # No repeats
        opt1 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
        )
        assert opt1.repeats_initial == 1
        assert opt1.repeats_surrogate == 1

        # Only repeats_initial
        opt2 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            repeats_initial=2,
        )
        assert opt2.repeats_initial == 2

        # Only repeats_surrogate
        opt3 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            repeats_surrogate=2,
        )
        assert opt3.repeats_surrogate == 2

        # Both
        opt4 = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            repeats_initial=2,
            repeats_surrogate=3,
        )
        assert opt4.repeats_initial == 2
        assert opt4.repeats_surrogate == 3

    def test_initial_design_repeats(self):
        """Test that initial design points are repeated correctly."""
        n_initial = 5
        repeats = 3

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=n_initial,
            repeats_initial=repeats,
            seed=42,
        )

        result = opt.optimize()

        # Should have at least n_initial * repeats evaluations
        assert result.nfev >= n_initial * repeats

        # Check that initial points are properly repeated
        # (first n_initial*repeats rows should have each unique point repeated)
        initial_evals = opt.X_[: n_initial * repeats]
        unique_initial = np.unique(initial_evals, axis=0)
        assert len(unique_initial) == n_initial

    def test_surrogate_repeats(self):
        """Test that surrogate-suggested points are repeated correctly."""
        n_initial = 3
        repeats_initial = 2
        repeats_surrogate = 3
        max_iter = 12  # 3*2 + 2*3 = 12 evaluations total

        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=max_iter,
            n_initial=n_initial,
            repeats_initial=repeats_initial,
            repeats_surrogate=repeats_surrogate,
            seed=42,
        )

        result = opt.optimize()

        # Should reach max_iter
        assert result.nfev == max_iter

        # Should have 2 sequential iterations (after initial design)
        assert result.nit == 2


class TestAggregatemeanVar:
    """Test suite for _aggregate_mean_var method."""

    def test_aggregate_mean_var_basic(self):
        """Test basic aggregation of mean and variance."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            repeats_initial=2,
        )

        X = np.array([[1, 2], [3, 4], [1, 2]])
        y = np.array([1.0, 2.0, 3.0])

        X_agg, y_mean, y_var = opt._aggregate_mean_var(X, y)

        # Should have 2 unique points
        assert X_agg.shape == (2, 2)

        # Check means
        np.testing.assert_array_equal(y_mean, np.array([2.0, 2.0]))

        # Check variances (population variance, ddof=0)
        np.testing.assert_array_equal(y_var, np.array([1.0, 0.0]))

    def test_aggregate_mean_var_multiple_repeats(self):
        """Test aggregation with multiple repeats."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            repeats_initial=3,
        )

        X = np.array([[1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2]])
        y = np.array([3.0, 3.0, 6.0, 6.0, 6.0, 6.0])

        X_agg, y_mean, y_var = opt._aggregate_mean_var(X, y)

        # Should have 2 unique points
        assert X_agg.shape == (2, 2)

        # Check means
        np.testing.assert_array_almost_equal(y_mean, np.array([4.0, 6.0]))

        # Check variances
        expected_var1 = np.var([3.0, 3.0, 6.0], ddof=0)
        expected_var2 = np.var([6.0, 6.0, 6.0], ddof=0)
        np.testing.assert_array_almost_equal(
            y_var, np.array([expected_var1, expected_var2])
        )

    def test_aggregate_mean_var_no_duplicates(self):
        """Test aggregation when there are no duplicates."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
        )

        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1.0, 2.0, 3.0])

        X_agg, y_mean, y_var = opt._aggregate_mean_var(X, y)

        # All points should be unique
        assert X_agg.shape == (3, 2)
        np.testing.assert_array_equal(y_mean, y)
        np.testing.assert_array_equal(
            y_var, np.zeros(3)
        )  # No variance with single observations


class TestUpdateStats:
    """Test suite for update_stats method."""

    def test_update_stats_without_noise(self):
        """Test update_stats for non-noisy optimization."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
        )

        opt.X_ = np.array([[1, 2], [3, 4], [0, 1]])
        opt.y_ = np.array([5.0, 25.0, 1.0])

        opt.update_stats()

        # Check basic stats
        assert opt.min_y == 1.0
        np.testing.assert_array_equal(opt.min_X, np.array([0, 1]))
        assert opt.counter == 3

        # Noise stats should be None
        assert opt.mean_X is None
        assert opt.mean_y is None
        assert opt.var_y is None

    def test_update_stats_with_noise(self):
        """Test update_stats for noisy optimization."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=3,
            repeats_initial=2,
        )

        # Simulated repeated evaluations
        opt.X_ = np.array([[1, 2], [1, 2], [3, 4], [3, 4], [0, 1], [0, 1]])
        opt.y_ = np.array([4.0, 6.0, 24.0, 26.0, 0.5, 1.5])

        opt.update_stats()

        # Check basic stats
        assert opt.min_y == 0.5
        np.testing.assert_array_equal(opt.min_X, np.array([0, 1]))
        assert opt.counter == 6

        # Check aggregated stats
        assert opt.mean_X is not None
        assert opt.mean_y is not None
        assert opt.var_y is not None

        # Should have 3 unique points
        assert opt.mean_X.shape == (3, 2)
        assert opt.mean_y.shape == (3,)
        assert opt.var_y.shape == (3,)

        # Check mean values
        expected_means = np.array([1.0, 5.0, 25.0])  # Sorted
        np.testing.assert_array_almost_equal(np.sort(opt.mean_y), expected_means)

        # Check best mean stats
        assert opt.min_mean_y == 1.0
        np.testing.assert_array_equal(opt.min_mean_X, np.array([0, 1]))

        # Variance of best point
        assert opt.min_var_y == np.var([0.5, 1.5], ddof=0)

    def test_update_stats_empty(self):
        """Test update_stats with empty arrays."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
        )

        opt.X_ = None
        opt.y_ = None

        # Should not raise error
        opt.update_stats()

        assert opt.min_y is None
        assert opt.min_X is None


class TestNoisyOptimizationIntegration:
    """Integration tests for full noisy optimization."""

    def test_noisy_optimization_completion(self):
        """Test that noisy optimization completes successfully."""

        def noisy_objective(X):
            return noisy_sphere(X, sigma=0.05, seed=42)

        opt = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=20,
            n_initial=5,
            repeats_initial=2,
            repeats_surrogate=2,
            seed=42,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success
        assert result.nfev == 20
        assert result.nit > 0  # Should perform some sequential iterations

        # Noise stats should be populated
        assert opt.mean_X is not None
        assert opt.mean_y is not None
        assert opt.var_y is not None
        assert opt.min_mean_y is not None

    def test_surrogate_fitted_on_aggregated_data(self):
        """Test that surrogate is fitted on aggregated mean values when noise=True."""

        def noisy_objective(X):
            return noisy_sphere(X, sigma=0.1, seed=123)

        opt = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=25,
            n_initial=5,
            repeats_initial=3,
            repeats_surrogate=2,
            seed=123,
            verbose=False,
        )

        result = opt.optimize()

        # With repeats, we should have fewer unique points than total evaluations
        n_unique = opt.mean_X.shape[0]
        n_total = opt.X_.shape[0]
        assert n_unique < n_total

        # Surrogate should be fitted on mean values
        # (can't directly test, but we verify optimization completes)
        assert result.success

    def test_deterministic_with_seed(self):
        """Test that optimization with noise is deterministic with seed."""

        def noisy_objective(X):
            return noisy_sphere(X, sigma=0.1)

        opt1 = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            repeats_initial=2,
            seed=42,
            verbose=False,
        )
        result1 = opt1.optimize()

        opt2 = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            repeats_initial=2,
            seed=42,
            verbose=False,
        )
        result2 = opt2.optimize()

        # Results should be similar (not exact due to noise in function)
        # But initial design should be identical
        np.testing.assert_array_almost_equal(opt1.X_[:10], opt2.X_[:10])

    def test_mixed_repeats(self):
        """Test optimization with different repeats for initial and surrogate."""

        def noisy_objective(X):
            return noisy_sphere(X, sigma=0.1, seed=99)

        n_initial = 3
        repeats_initial = 3
        repeats_surrogate = 2
        max_iter = 15  # 3*3 + 3*2 = 15

        opt = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=max_iter,
            n_initial=n_initial,
            repeats_initial=repeats_initial,
            repeats_surrogate=repeats_surrogate,
            seed=99,
            verbose=False,
        )

        result = opt.optimize()

        assert result.success
        assert result.nfev == max_iter

        # Should have 3 sequential iterations
        assert result.nit == 3

    def test_convergence_with_noise(self):
        """Test that optimizer converges towards optimum despite noise."""

        def noisy_objective(X):
            return noisy_sphere(X, sigma=0.01, seed=77)

        opt = SpotOptim(
            fun=noisy_objective,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=10,
            repeats_initial=2,
            repeats_surrogate=2,
            seed=77,
            verbose=False,
        )

        result = opt.optimize()

        # With low noise, should find a good solution near origin
        assert result.fun < 1.0  # Should be close to global optimum at (0, 0)
        assert np.linalg.norm(result.x) < 2.0  # Solution should be near origin
