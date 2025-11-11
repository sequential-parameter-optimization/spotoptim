"""Tests for Kriging surrogate model."""

import pytest
import numpy as np
from spotoptim.surrogate import Kriging
from spotoptim import SpotOptim


class TestKriging:
    """Test suite for Kriging surrogate model."""

    def test_kriging_initialization(self):
        """Test Kriging initialization."""
        model = Kriging()
        assert model.kernel == "gauss"
        assert model.n_theta is None
        assert model.min_theta == -3.0
        assert model.max_theta == 2.0

    def test_kriging_fit_predict(self):
        """Test basic fit and predict."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.25, 1.0])

        model = Kriging(seed=42)
        model.fit(X, y)

        # Check fitted attributes
        assert model.X_ is not None
        assert model.y_ is not None
        assert model.theta_ is not None
        assert model.mu_ is not None
        assert model.sigma2_ is not None

        # Test prediction
        X_test = np.array([[0.25], [0.75]])
        y_pred = model.predict(X_test)

        assert y_pred.shape == (2,)
        assert np.all(np.isfinite(y_pred))

    def test_kriging_predict_with_std(self):
        """Test prediction with standard deviations."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.25, 1.0])

        model = Kriging(seed=42)
        model.fit(X, y)

        X_test = np.array([[0.25], [0.75]])
        y_pred, y_std = model.predict(X_test, return_std=True)

        assert y_pred.shape == (2,)
        assert y_std.shape == (2,)
        assert np.all(y_std >= 0)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    def test_kriging_2d(self):
        """Test Kriging on 2D problem."""
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [0.5, 0.0], [0.0, 0.5]])
        y = np.sum(X**2, axis=1)

        model = Kriging(seed=42)
        model.fit(X, y)

        X_test = np.array([[0.25, 0.25], [0.75, 0.75]])
        y_pred = model.predict(X_test)

        assert y_pred.shape == (2,)
        assert np.all(np.isfinite(y_pred))

    def test_spotoptim_with_kriging(self):
        """Test SpotOptim with Kriging surrogate."""

        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        # Create Kriging surrogate
        kriging = Kriging(seed=42)

        # Create optimizer with Kriging surrogate
        optimizer = SpotOptim(
            fun=sphere,
            bounds=bounds,
            max_iter=10,
            n_initial=5,
            surrogate=kriging,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        # Check results
        assert result.success is True
        assert result.nfev == 10  # max_iter now includes initial design
        assert result.nit == 5  # 10 total - 5 initial = 5 sequential iterations
        assert len(result.x) == 2

        # Solution should be close to [0, 0]
        assert result.fun < 5.0  # Reasonable result

    def test_spotoptim_kriging_vs_gp(self):
        """Compare SpotOptim with Kriging vs default GP."""

        def rosenbrock(X):
            X = np.atleast_2d(X)
            x = X[:, 0]
            y = X[:, 1]
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        bounds = [(-2, 2), (-2, 2)]

        # With Kriging
        kriging = Kriging(seed=42)
        optimizer_kriging = SpotOptim(
            fun=rosenbrock,
            bounds=bounds,
            max_iter=15,
            n_initial=8,
            surrogate=kriging,
            seed=42,
            verbose=False,
        )
        result_kriging = optimizer_kriging.optimize()

        # With default GP
        optimizer_gp = SpotOptim(
            fun=rosenbrock,
            bounds=bounds,
            max_iter=15,
            n_initial=8,
            seed=42,
            verbose=False,
        )
        result_gp = optimizer_gp.optimize()

        # Both should succeed
        assert result_kriging.success is True
        assert result_gp.success is True

        # Both should find reasonable solutions
        assert result_kriging.fun < 100.0
        assert result_gp.fun < 100.0

    def test_kriging_custom_parameters(self):
        """Test Kriging with custom parameters."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.25, 1.0])

        model = Kriging(noise=1e-4, min_theta=-2.0, max_theta=3.0, seed=123)
        model.fit(X, y)

        X_test = np.array([[0.25]])
        y_pred = model.predict(X_test)

        assert np.isfinite(y_pred[0])

    def test_kriging_input_validation(self):
        """Test input validation."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.25, 1.0])

        model = Kriging(seed=42)
        model.fit(X, y)

        # Wrong number of features
        with pytest.raises(ValueError):
            model.predict(np.array([[0.25, 0.5]]))

    def test_kriging_seed_reproducibility(self):
        """Test that same seed produces same results."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.25, 1.0])

        model1 = Kriging(seed=42)
        model1.fit(X, y)
        pred1 = model1.predict(np.array([[0.25]]))

        model2 = Kriging(seed=42)
        model2.fit(X, y)
        pred2 = model2.predict(np.array([[0.25]]))

        np.testing.assert_array_almost_equal(pred1, pred2)
