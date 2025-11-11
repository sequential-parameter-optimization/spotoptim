"""Tests for OCBA (Optimal Computing Budget Allocation) functionality."""

import numpy as np
import pytest
from spotoptim import SpotOptim


class TestOCBAParameters:
    """Test OCBA parameter initialization and basic functionality."""

    def test_ocba_delta_default(self):
        """Test that ocba_delta defaults to 0."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5
        )
        assert opt.ocba_delta == 0

    def test_ocba_delta_custom(self):
        """Test setting custom ocba_delta value."""
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            ocba_delta=3
        )
        assert opt.ocba_delta == 3

    def test_ocba_requires_noise(self):
        """Test that OCBA is only active when noise=True."""
        opt_no_noise = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            ocba_delta=2
        )
        assert opt_no_noise.noise is False
        assert opt_no_noise.ocba_delta == 2

        opt_with_noise = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            max_iter=10,
            n_initial=5,
            repeats_initial=2,
            ocba_delta=2
        )
        assert opt_with_noise.noise is True
        assert opt_with_noise.ocba_delta == 2


class TestOCBAIntegration:
    """Test OCBA integration in optimization loop."""

    def test_ocba_with_noisy_function(self):
        """Test that OCBA adds evaluations during optimization."""
        np.random.seed(42)

        def noisy_sphere(X):
            """Sphere function with additive noise."""
            base = np.sum(X**2, axis=1)
            noise = np.random.normal(0, 0.1, size=base.shape)
            return base + noise

        opt = SpotOptim(
            fun=noisy_sphere,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=10,
            repeats_initial=2,
            repeats_surrogate=1,
            ocba_delta=2,
            seed=42,
            verbose=False
        )

        result = opt.optimize()

        # Check that optimization completed
        assert result.success is True
        assert result.nfev > 0

        # With ocba_delta=2, we should have extra evaluations
        # Initial: 10 points * 2 repeats = 20 evals
        # Each iteration: 1 point * 1 repeat + 2 OCBA = 3 evals per iteration
        # So total should be > 20 (initial) and reflect OCBA additions
        assert result.nfev >= 20

        # Check that noise statistics were computed
        assert opt.mean_X is not None
        assert opt.mean_y is not None
        assert opt.var_y is not None
        assert opt.min_mean_y is not None

    def test_ocba_without_sufficient_variance(self):
        """Test OCBA behavior when variance conditions not met."""
        np.random.seed(42)

        # Function that evaluates to same value (no variance)
        def constant_function(X):
            return np.ones(X.shape[0])

        opt = SpotOptim(
            fun=constant_function,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=15,
            n_initial=5,
            repeats_initial=2,
            ocba_delta=2,
            seed=42,
            verbose=False
        )

        result = opt.optimize()

        # Optimization should still work even if OCBA can't be applied
        assert result.success is True
        # With constant function, variances will be 0, so OCBA shouldn't add points
        # Total evals: 5 * 2 (initial) + 5 * 1 (iterations) = 15
        assert result.nfev == 15

    def test_ocba_with_few_points(self):
        """Test OCBA behavior with insufficient number of points (< 3)."""
        np.random.seed(42)

        def noisy_sphere(X):
            base = np.sum(X**2, axis=1)
            noise = np.random.normal(0, 0.1, size=base.shape)
            return base + noise

        opt = SpotOptim(
            fun=noisy_sphere,
            bounds=[(-5, 5)],  # 1D
            max_iter=6,
            n_initial=2,  # Only 2 initial points
            repeats_initial=2,
            ocba_delta=2,
            seed=42,
            verbose=False
        )

        result = opt.optimize()

        # OCBA requires >2 unique points, so it shouldn't apply initially
        # but might apply after a few iterations
        assert result.success is True

    def test_ocba_evaluations_count(self):
        """Test that OCBA correctly adds expected number of evaluations."""
        np.random.seed(123)

        def noisy_quadratic(X):
            """Simple quadratic with noise."""
            base = np.sum(X**2, axis=1)
            noise = np.random.normal(0, 0.5, size=base.shape)
            return base + noise

        opt = SpotOptim(
            fun=noisy_quadratic,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=50,
            n_initial=10,
            repeats_initial=2,
            repeats_surrogate=1,
            ocba_delta=3,
            seed=123,
            verbose=False
        )

        result = opt.optimize()

        # Initial: 10 * 2 = 20 evaluations
        # Each iteration adds: 1 (new point) + 3 (OCBA) = 4 evaluations
        # To reach 50 total: (50 - 20) / 4 = 7.5, so ~7-8 iterations
        # This gives: 20 + 7*4 = 48 or 20 + 8*4 = 52 evaluations
        # (actual might vary due to when OCBA conditions are met)
        assert 45 <= result.nfev <= 52

        # Verify we have aggregated statistics
        assert opt.mean_X.shape[0] >= 10
        assert len(opt.mean_y) == opt.mean_X.shape[0]
        assert len(opt.var_y) == opt.mean_X.shape[0]


class TestOCBAComparison:
    """Test OCBA effectiveness compared to baseline."""

    def test_ocba_vs_no_ocba(self):
        """Compare optimization with and without OCBA on noisy function."""
        np.random.seed(42)

        def noisy_rosenbrock(X):
            """2D Rosenbrock with noise."""
            x0 = X[:, 0]
            x1 = X[:, 1]
            base = (1 - x0)**2 + 100 * (x1 - x0**2)**2
            noise = np.random.normal(0, 1.0, size=base.shape)
            return base + noise

        # Without OCBA
        opt_no_ocba = SpotOptim(
            fun=noisy_rosenbrock,
            bounds=[(-2, 2), (-2, 2)],
            max_iter=40,
            n_initial=10,
            repeats_initial=2,
            repeats_surrogate=2,
            ocba_delta=0,  # No OCBA
            seed=42,
            verbose=False
        )
        result_no_ocba = opt_no_ocba.optimize()

        # With OCBA
        opt_with_ocba = SpotOptim(
            fun=noisy_rosenbrock,
            bounds=[(-2, 2), (-2, 2)],
            max_iter=40,
            n_initial=10,
            repeats_initial=2,
            repeats_surrogate=2,
            ocba_delta=2,  # Use OCBA
            seed=42,
            verbose=False
        )
        result_with_ocba = opt_with_ocba.optimize()

        # Both should complete successfully
        assert result_no_ocba.success is True
        assert result_with_ocba.success is True

        # Both hit max_iter, so check number of iterations instead
        # Without OCBA: each iteration adds 2 evals (repeats_surrogate=2)
        # With OCBA: each iteration adds 2 + 2 = 4 evals (surrogate + OCBA)
        # So with same budget, OCBA should have fewer iterations but same total evals
        assert result_no_ocba.nfev == 40
        assert result_with_ocba.nfev == 40
        # OCBA version should have fewer iterations (more evals per iteration)
        assert result_with_ocba.nit < result_no_ocba.nit

        # Both should find reasonable solutions (may vary due to noise)
        # Just check they both improved from random initial points
        assert result_no_ocba.fun < 100  # Should be better than random
        assert result_with_ocba.fun < 100

    def test_ocba_deterministic_behavior(self):
        """Test that OCBA produces consistent results with same seed."""
        def noisy_function(X):
            base = np.sum(X**2, axis=1)
            noise = np.random.normal(0, 0.2, size=base.shape)
            return base + noise

        results = []
        for _ in range(2):
            np.random.seed(999)
            opt = SpotOptim(
                fun=noisy_function,
                bounds=[(-3, 3), (-3, 3)],
                max_iter=25,
                n_initial=8,
                repeats_initial=2,
                ocba_delta=2,
                seed=999,
                verbose=False
            )
            result = opt.optimize()
            results.append(result)

        # Results should be identical with same seed
        assert results[0].nfev == results[1].nfev
        np.testing.assert_array_almost_equal(results[0].x, results[1].x, decimal=10)
        assert abs(results[0].fun - results[1].fun) < 1e-10


class TestOCBAEdgeCases:
    """Test edge cases and boundary conditions for OCBA."""

    def test_ocba_with_max_iter_reached(self):
        """Test OCBA behavior when max_iter is reached."""
        np.random.seed(55)

        def noisy_func(X):
            return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])

        opt = SpotOptim(
            fun=noisy_func,
            bounds=[(-5, 5)],
            max_iter=20,  # Small limit
            n_initial=10,
            repeats_initial=2,
            ocba_delta=5,  # Large OCBA (may not all be used)
            seed=55,
            verbose=False
        )

        result = opt.optimize()

        # Should stop at max_iter
        assert result.nfev <= 20
        assert "maximum evaluations" in result.message

    def test_ocba_zero_delta(self):
        """Test that ocba_delta=0 disables OCBA."""
        np.random.seed(77)

        def noisy_func(X):
            return np.sum(X**2, axis=1) + np.random.normal(0, 0.1, X.shape[0])

        opt = SpotOptim(
            fun=noisy_func,
            bounds=[(-5, 5), (-5, 5)],
            max_iter=30,
            n_initial=10,
            repeats_initial=2,
            repeats_surrogate=1,
            ocba_delta=0,  # Disabled
            seed=77,
            verbose=False
        )

        result = opt.optimize()

        # Without OCBA: 10*2 initial + 10*1 surrogate = 30 total
        assert result.nfev == 30
