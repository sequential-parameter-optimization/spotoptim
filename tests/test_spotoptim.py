import pytest
import numpy as np
from scipy.optimize import OptimizeResult
from spotoptim.SpotOptim import SpotOptim


class TestSpotOptimOptimize:
    """Test suite for the SpotOptim.optimize() method."""

    def test_optimize_simple_quadratic(self):
        """Test optimization on a simple quadratic function."""

        def quadratic(X):
            """Simple quadratic: f(x) = x^2"""
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5)]
        optimizer = SpotOptim(
            fun=quadratic,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Check result type
        assert isinstance(result, OptimizeResult)

        # Check that best point is near optimum
        assert np.abs(result.x[0]) < 1.0, f"Expected x near 0, got {result.x[0]}"

        # Check that function value is small
        assert result.fun < 1.0, f"Expected f(x) near 0, got {result.fun}"

        # Check result attributes
        assert result.success is True
        assert result.nfev == 15  # 5 initial + 10 iterations
        assert result.nit == 10

    def test_optimize_rosenbrock_2d(self):
        """Test optimization on 2D Rosenbrock function."""

        def rosenbrock(X):
            """Rosenbrock function."""
            X = np.atleast_2d(X)
            x = X[:, 0]
            y = X[:, 1]
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        bounds = [(-2, 2), (-2, 2)]
        optimizer = SpotOptim(
            fun=rosenbrock,
            bounds=bounds,
            max_iter=20,
            n_initial=10,
            acquisition="ei",
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        assert result.success is True
        assert result.nfev == 30  # 10 initial + 20 iterations
        assert result.nit == 20
        assert len(result.x) == 2

        # Check that result is reasonable (Rosenbrock optimum is at [1, 1])
        assert result.fun < 100.0, f"Function value too high: {result.fun}"

    def test_optimize_with_custom_initial_design(self):
        """Test optimization with user-provided initial design."""

        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        # Custom initial design
        X0 = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, -2.0], [-2.0, 2.0], [0.5, 0.5]])

        optimizer = SpotOptim(
            fun=sphere, bounds=bounds, max_iter=5, n_initial=5, seed=42, verbose=False
        )

        result = optimizer.optimize(X0=X0)

        assert isinstance(result, OptimizeResult)
        assert result.nfev == 10  # 5 initial + 5 iterations
        assert result.nit == 5

    def test_optimize_different_acquisitions(self):
        """Test optimization with different acquisition functions."""

        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-3, 3), (-3, 3)]

        for acquisition in ["ei", "y", "pi"]:
            optimizer = SpotOptim(
                fun=sphere,
                bounds=bounds,
                max_iter=5,
                n_initial=5,
                acquisition=acquisition,
                seed=42,
                verbose=False,
            )

            result = optimizer.optimize()

            assert isinstance(result, OptimizeResult)
            assert result.success is True
            assert (
                result.fun < 10.0
            ), f"Acquisition '{acquisition}' produced poor result: {result.fun}"

    def test_optimize_single_dimension(self):
        """Test optimization on a 1D function."""

        def parabola(X):
            X = np.atleast_2d(X)
            return (X[:, 0] - 3) ** 2

        bounds = [(0, 10)]

        optimizer = SpotOptim(
            fun=parabola,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        assert len(result.x) == 1
        # Optimum is at x=3
        assert np.abs(result.x[0] - 3.0) < 1.0, f"Expected x near 3, got {result.x[0]}"

    def test_optimize_high_dimension(self):
        """Test optimization on a higher-dimensional function."""

        def sphere_nd(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        n_dim = 5
        bounds = [(-5, 5)] * n_dim

        optimizer = SpotOptim(
            fun=sphere_nd,
            bounds=bounds,
            max_iter=15,
            n_initial=10,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        assert len(result.x) == n_dim
        assert result.nfev == 25  # 10 initial + 15 iterations

    def test_optimize_result_attributes(self):
        """Test that all expected result attributes are present."""

        def simple_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-2, 2)]

        optimizer = SpotOptim(
            fun=simple_func,
            bounds=bounds,
            max_iter=5,
            n_initial=3,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Check all expected attributes
        assert hasattr(result, "x")
        assert hasattr(result, "fun")
        assert hasattr(result, "nfev")
        assert hasattr(result, "nit")
        assert hasattr(result, "success")
        assert hasattr(result, "message")
        assert hasattr(result, "X")
        assert hasattr(result, "y")

        # Check types
        assert isinstance(result.x, np.ndarray)
        assert isinstance(result.fun, (float, np.floating))
        assert isinstance(result.nfev, (int, np.integer))
        assert isinstance(result.nit, (int, np.integer))
        assert isinstance(result.success, bool)
        assert isinstance(result.message, str)
        assert isinstance(result.X, np.ndarray)
        assert isinstance(result.y, np.ndarray)

    def test_optimize_stores_all_evaluations(self):
        """Test that optimizer stores all function evaluations."""

        def counter_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]
        n_initial = 5
        max_iter = 10

        optimizer = SpotOptim(
            fun=counter_func,
            bounds=bounds,
            max_iter=max_iter,
            n_initial=n_initial,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Check that all evaluations are stored
        total_evals = n_initial + max_iter
        assert len(result.X) == total_evals
        assert len(result.y) == total_evals
        assert result.X.shape[0] == total_evals
        assert result.X.shape[1] == 2  # 2D problem

    def test_optimize_best_is_minimum(self):
        """Test that the best result is indeed the minimum found."""

        def test_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-3, 3), (-3, 3)]

        optimizer = SpotOptim(
            fun=test_func,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Best value should be minimum of all evaluations
        assert result.fun == np.min(result.y)

        # Best point should correspond to minimum value
        best_idx = np.argmin(result.y)
        np.testing.assert_array_almost_equal(result.x, result.X[best_idx])

    def test_optimize_with_seed_reproducibility(self):
        """Test that same seed produces reproducible results."""

        def test_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]
        seed = 123

        # Run optimization twice with same seed
        optimizer1 = SpotOptim(
            fun=test_func,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            seed=seed,
            verbose=False,
        )
        result1 = optimizer1.optimize()

        optimizer2 = SpotOptim(
            fun=test_func,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            seed=seed,
            verbose=False,
        )
        result2 = optimizer2.optimize()

        # Results should be identical
        np.testing.assert_array_almost_equal(result1.x, result2.x)
        np.testing.assert_almost_equal(result1.fun, result2.fun)
        np.testing.assert_array_almost_equal(result1.X, result2.X)
        np.testing.assert_array_almost_equal(result1.y, result2.y)

    def test_optimize_verbose_mode(self, capsys):
        """Test that verbose mode produces output."""

        def test_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-2, 2)]

        optimizer = SpotOptim(
            fun=test_func, bounds=bounds, max_iter=3, n_initial=2, seed=42, verbose=True
        )

        result = optimizer.optimize()

        captured = capsys.readouterr()

        # Check that some output was produced
        assert len(captured.out) > 0
        assert "Initial best" in captured.out or "Iteration" in captured.out

    def test_optimize_with_integer_variables(self):
        """Test optimization with integer variable types."""

        def discrete_func(X):
            X = np.atleast_2d(X)
            return np.sum(np.round(X) ** 2, axis=1)

        bounds = [(-5, 5), (-5, 5)]
        var_type = ["int", "int"]

        optimizer = SpotOptim(
            fun=discrete_func,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            var_type=var_type,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        # With integer variables, the result should be close to integers
        assert np.allclose(result.x, np.round(result.x), atol=1e-6)

    def test_optimize_mixed_variable_types(self):
        """Test optimization with mixed continuous and integer variables."""

        def mixed_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5), (-5, 5)]
        var_type = ["num", "int", "float"]

        optimizer = SpotOptim(
            fun=mixed_func,
            bounds=bounds,
            max_iter=8,
            n_initial=5,
            var_type=var_type,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        assert len(result.x) == 3
        # Second variable should be integer
        assert np.isclose(result.x[1], np.round(result.x[1]), atol=1e-6)

    def test_optimize_tolerance_x(self):
        """Test that tolerance_x parameter is respected."""

        def test_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-2, 2), (-2, 2)]

        optimizer = SpotOptim(
            fun=test_func,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            tolerance_x=0.5,  # Large tolerance
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        # With large tolerance, points should be more spread out
        # This is a soft check
        assert result.nfev <= 15

    def test_optimize_zero_iterations(self):
        """Test optimization with zero iterations (only initial design)."""

        def test_func(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5)]

        optimizer = SpotOptim(
            fun=test_func,
            bounds=bounds,
            max_iter=0,
            n_initial=5,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        assert result.nfev == 5  # Only initial design
        assert result.nit == 0
        assert len(result.X) == 5

    def test_optimize_beale_function(self):
        """Test on Beale function, a common test function."""

        def beale(X):
            """Beale function with optimum at (3, 0.5)."""
            X = np.atleast_2d(X)
            x = X[:, 0]
            y = X[:, 1]
            term1 = (1.5 - x + x * y) ** 2
            term2 = (2.25 - x + x * y**2) ** 2
            term3 = (2.625 - x + x * y**3) ** 2
            return term1 + term2 + term3

        bounds = [(-4.5, 4.5), (-4.5, 4.5)]

        optimizer = SpotOptim(
            fun=beale, bounds=bounds, max_iter=20, n_initial=10, seed=42, verbose=False
        )

        result = optimizer.optimize()

        assert isinstance(result, OptimizeResult)
        assert result.success is True
        # Beale function has a very narrow valley, so we just check it found something reasonable
        assert result.fun < 100.0
