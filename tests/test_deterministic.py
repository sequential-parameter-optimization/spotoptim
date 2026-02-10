# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
from spotoptim.SpotOptim import SpotOptim


class TestDeterministicBehavior:
    """Test suite for deterministic and non-deterministic behavior of SpotOptim."""

    def test_generate_initial_design_without_seed_is_non_deterministic(self):
        """Test that _generate_initial_design() produces different results without seed."""

        def dummy_fun(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        # Create two optimizers without seed
        opt1 = SpotOptim(fun=dummy_fun, bounds=bounds, n_initial=10)
        opt2 = SpotOptim(fun=dummy_fun, bounds=bounds, n_initial=10)

        # Generate initial designs
        X1 = opt1._generate_initial_design()
        X2 = opt2._generate_initial_design()

        # They should be different (with very high probability)
        assert not np.allclose(
            X1, X2
        ), "Expected different designs without seed, but got identical results"

        # Check shapes are correct
        assert X1.shape == (10, 2)
        assert X2.shape == (10, 2)

        # Check bounds are respected
        assert np.all(X1 >= -5) and np.all(X1 <= 5)
        assert np.all(X2 >= -5) and np.all(X2 <= 5)

    def test_generate_initial_design_with_seed_is_deterministic(self):
        """Test that _generate_initial_design() produces identical results with same seed."""

        def dummy_fun(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]
        seed = 42

        # Create two optimizers with same seed
        opt1 = SpotOptim(fun=dummy_fun, bounds=bounds, n_initial=10, seed=seed)
        opt2 = SpotOptim(fun=dummy_fun, bounds=bounds, n_initial=10, seed=seed)

        # Generate initial designs
        X1 = opt1._generate_initial_design()
        X2 = opt2._generate_initial_design()

        # They should be identical
        assert np.allclose(
            X1, X2
        ), "Expected identical designs with same seed, but got different results"

        # Verify they are exactly equal
        np.testing.assert_array_equal(X1, X2)

    def test_generate_initial_design_different_seeds_produce_different_results(self):
        """Test that different seeds produce different initial designs."""

        def dummy_fun(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        # Create optimizers with different seeds
        opt1 = SpotOptim(fun=dummy_fun, bounds=bounds, n_initial=10, seed=42)
        opt2 = SpotOptim(fun=dummy_fun, bounds=bounds, n_initial=10, seed=123)

        # Generate initial designs
        X1 = opt1._generate_initial_design()
        X2 = opt2._generate_initial_design()

        # They should be different
        assert not np.allclose(
            X1, X2
        ), "Expected different designs with different seeds"

    def test_multiple_calls_with_same_seed_are_not_deterministic(self):
        """Test that multiple calls to _generate_initial_design() with same optimizer are different.

        This tests that the internal state of the LHS sampler advances with each call.
        """

        def dummy_fun(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        # Create one optimizer with seed
        opt = SpotOptim(fun=dummy_fun, bounds=bounds, n_initial=10, seed=42)

        # Generate initial designs twice
        X1 = opt._generate_initial_design()
        X2 = opt._generate_initial_design()

        # They should be different (sampler state advances)
        assert not np.allclose(
            X1, X2
        ), "Expected different designs on multiple calls to same optimizer"

    def test_optimize_with_seed_is_deterministic(self):
        """Test that full optimization is deterministic with seed."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]
        seed = 42

        # Run optimization twice with same seed
        opt1 = SpotOptim(fun=sphere, bounds=bounds, max_iter=5, n_initial=5, seed=seed)
        result1 = opt1.optimize()

        opt2 = SpotOptim(fun=sphere, bounds=bounds, max_iter=5, n_initial=5, seed=seed)
        result2 = opt2.optimize()

        # Results should be identical
        np.testing.assert_array_almost_equal(result1.x, result2.x)
        assert np.isclose(result1.fun, result2.fun)

        # All evaluated points should be identical
        np.testing.assert_array_almost_equal(opt1.X_, opt2.X_)
        np.testing.assert_array_almost_equal(opt1.y_, opt2.y_)

    def test_optimize_without_seed_is_non_deterministic(self):
        """Test that full optimization produces different results without seed."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        # Run optimization twice without seed
        opt1 = SpotOptim(fun=sphere, bounds=bounds, max_iter=5, n_initial=5)
        _ = opt1.optimize()

        opt2 = SpotOptim(fun=sphere, bounds=bounds, max_iter=5, n_initial=5)
        _ = opt2.optimize()

        # Initial designs should be different
        assert not np.allclose(
            opt1.X_[:5], opt2.X_[:5]
        ), "Expected different initial designs without seed"

    def test_seed_affects_surrogate_model(self):
        """Test that seed affects the Gaussian Process surrogate model."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        # Create two optimizers with same seed
        opt1 = SpotOptim(fun=sphere, bounds=bounds, max_iter=5, n_initial=5, seed=42)
        opt2 = SpotOptim(fun=sphere, bounds=bounds, max_iter=5, n_initial=5, seed=42)

        # Check that surrogate models have same random state
        assert opt1.surrogate.random_state == opt2.surrogate.random_state == 42

    def test_deterministic_with_provided_initial_design(self):
        """Test deterministic behavior when providing custom initial design."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]
        seed = 42

        # Provide same initial design
        X0 = np.array([[-2.0, -2.0], [2.0, 2.0], [0.0, 0.0]])
        n_initial = X0.shape[0]  # 3 points

        opt1 = SpotOptim(
            fun=sphere, bounds=bounds, max_iter=10, n_initial=n_initial, seed=seed
        )
        result1 = opt1.optimize(X0=X0.copy())

        opt2 = SpotOptim(
            fun=sphere, bounds=bounds, max_iter=10, n_initial=n_initial, seed=seed
        )
        result2 = opt2.optimize(X0=X0.copy())

        # Results should be identical
        np.testing.assert_array_almost_equal(result1.x, result2.x)
        np.testing.assert_array_almost_equal(opt1.X_, opt2.X_)

    def test_seed_parameter_types(self):
        """Test that different seed types work correctly."""

        def dummy_fun(X):
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5)]

        # Integer seed
        opt1 = SpotOptim(fun=dummy_fun, bounds=bounds, seed=42)
        assert opt1.seed == 42

        # None seed (default)
        opt2 = SpotOptim(fun=dummy_fun, bounds=bounds, seed=None)
        assert opt2.seed is None

        # Zero seed
        opt3 = SpotOptim(fun=dummy_fun, bounds=bounds, seed=0)
        assert opt3.seed == 0
        X1 = opt3._generate_initial_design()

        opt4 = SpotOptim(fun=dummy_fun, bounds=bounds, seed=0)
        X2 = opt4._generate_initial_design()

        # Same seed (0) should produce same results
        np.testing.assert_array_equal(X1, X2)

    def test_reproducibility_across_dimensions(self):
        """Test that seeded behavior is reproducible across different problem dimensions."""

        def sphere(X):
            return np.sum(X**2, axis=1)

        seed = 42

        # 2D problem
        opt_2d_a = SpotOptim(
            fun=sphere, bounds=[(-5, 5), (-5, 5)], n_initial=5, seed=seed
        )
        X_2d_a = opt_2d_a._generate_initial_design()

        opt_2d_b = SpotOptim(
            fun=sphere, bounds=[(-5, 5), (-5, 5)], n_initial=5, seed=seed
        )
        X_2d_b = opt_2d_b._generate_initial_design()

        np.testing.assert_array_equal(X_2d_a, X_2d_b)

        # 3D problem
        opt_3d_a = SpotOptim(
            fun=sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)], n_initial=5, seed=seed
        )
        X_3d_a = opt_3d_a._generate_initial_design()

        opt_3d_b = SpotOptim(
            fun=sphere, bounds=[(-5, 5), (-5, 5), (-5, 5)], n_initial=5, seed=seed
        )
        X_3d_b = opt_3d_b._generate_initial_design()

        np.testing.assert_array_equal(X_3d_a, X_3d_b)

        # But 2D and 3D should be different (different samplers)
        assert X_2d_a.shape != X_3d_a.shape
