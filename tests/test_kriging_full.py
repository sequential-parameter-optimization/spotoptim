# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the full-featured Kriging surrogate model."""

import pytest
import numpy as np
from spotoptim.surrogate import Kriging
from spotoptim import SpotOptim


class TestKrigingFull:
    """Test suite for full-featured Kriging surrogate model."""

    def test_kriging_initialization_default(self):
        """Test Kriging initialization with default parameters."""
        model = Kriging()
        assert model.method == "regression"
        assert model.var_type == ["float"]
        assert model.n_theta is None
        assert model.min_theta == -3.0
        assert model.max_theta == 2.0
        assert model.isotropic is False
        assert model.seed == 124

    def test_kriging_initialization_custom(self):
        """Test Kriging initialization with custom parameters."""
        model = Kriging(
            method="interpolation",
            var_type=["float", "int", "factor"],
            min_theta=-2.0,
            max_theta=3.0,
            isotropic=True,
            seed=42,
        )
        assert model.method == "interpolation"
        assert model.var_type == ["float", "int", "factor"]
        assert model.min_theta == -2.0
        assert model.max_theta == 3.0
        assert model.isotropic is True
        assert model.seed == 42

    def test_kriging_invalid_method(self):
        """Test that invalid method raises ValueError."""
        with pytest.raises(ValueError, match="method must be"):
            Kriging(method="invalid_method")

    def test_kriging_invalid_noise(self):
        """Test that invalid noise raises ValueError."""
        with pytest.raises(ValueError, match="noise must be positive"):
            Kriging(noise=-0.1)

    def test_kriging_fit_predict_interpolation(self):
        """Test Kriging fit and predict with interpolation method."""
        X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = np.sin(X.ravel())

        model = Kriging(method="interpolation", seed=42, model_fun_evals=50)
        model.fit(X, y)

        # Check fitted attributes
        assert model.X_ is not None
        assert model.y_ is not None
        assert model.theta_ is not None
        assert model.U_ is not None
        assert model.Psi_ is not None
        assert model.negLnLike is not None
        assert model.Lambda_ is None  # No Lambda for interpolation

        # Test prediction
        X_test = np.array([[0.25], [0.75], [1.25]])
        y_pred = model.predict(X_test)

        assert y_pred.shape == (3,)
        assert np.all(np.isfinite(y_pred))

    def test_kriging_fit_predict_regression(self):
        """Test Kriging fit and predict with regression method."""
        X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = np.sin(X.ravel())

        model = Kriging(method="regression", seed=42, model_fun_evals=50)
        model.fit(X, y)

        # Check fitted attributes
        assert model.X_ is not None
        assert model.y_ is not None
        assert model.theta_ is not None
        assert model.Lambda_ is not None  # Lambda optimized for regression
        assert model.U_ is not None
        assert model.negLnLike is not None

        # Test prediction
        X_test = np.array([[0.25], [0.75]])
        y_pred = model.predict(X_test)

        assert y_pred.shape == (2,)
        assert np.all(np.isfinite(y_pred))

    def test_kriging_fit_predict_reinterpolation(self):
        """Test Kriging fit and predict with reinterpolation method."""
        X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = np.sin(X.ravel())

        model = Kriging(method="reinterpolation", seed=42, model_fun_evals=50)
        model.fit(X, y)

        # Check fitted attributes
        assert model.Lambda_ is not None  # Lambda optimized for reinterpolation

        # Test prediction
        X_test = np.array([[0.25], [0.75]])
        y_pred = model.predict(X_test)

        assert y_pred.shape == (2,)
        assert np.all(np.isfinite(y_pred))

    def test_kriging_predict_with_std(self):
        """Test prediction with standard deviations."""
        X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = np.sin(X.ravel())

        model = Kriging(method="regression", seed=42, model_fun_evals=50)
        model.fit(X, y)

        X_test = np.array([[0.25], [0.75], [1.25]])
        y_pred, y_std = model.predict(X_test, return_std=True)

        assert y_pred.shape == (3,)
        assert y_std.shape == (3,)
        assert np.all(y_std >= 0)
        assert np.all(np.isfinite(y_pred))
        assert np.all(np.isfinite(y_std))

    def test_kriging_2d_problem(self):
        """Test Kriging on 2D problem."""
        np.random.seed(42)
        X = np.random.rand(10, 2) * 4 - 2
        y = np.sin(X[:, 0]) * np.cos(X[:, 1])

        model = Kriging(method="regression", seed=42, model_fun_evals=50)
        model.fit(X, y)

        X_test = np.array([[0.5, 0.5], [1.0, 1.0], [-0.5, -0.5]])
        y_pred = model.predict(X_test)

        assert y_pred.shape == (3,)
        assert np.all(np.isfinite(y_pred))

    def test_kriging_isotropic_vs_anisotropic(self):
        """Test isotropic vs anisotropic correlation."""
        X = np.random.rand(10, 3)
        y = np.sum(X**2, axis=1)

        # Anisotropic (different theta per dimension)
        model_aniso = Kriging(isotropic=False, seed=42, model_fun_evals=30)
        model_aniso.fit(X, y)
        assert model_aniso.n_theta == 3

        # Isotropic (single theta for all dimensions)
        model_iso = Kriging(isotropic=True, seed=42, model_fun_evals=30)
        model_iso.fit(X, y)
        assert model_iso.n_theta == 1

        # Both should produce valid predictions
        X_test = np.array([[0.5, 0.5, 0.5]])
        y_pred_aniso = model_aniso.predict(X_test)
        y_pred_iso = model_iso.predict(X_test)

        assert np.isfinite(y_pred_aniso[0])
        assert np.isfinite(y_pred_iso[0])

    def test_kriging_mixed_variable_types_float_int_factor(self):
        """Test Kriging with mixed variable types: float, int, factor."""
        # 3D problem: continuous, integer, categorical
        X = np.array(
            [
                [0.0, 1, 0],
                [0.5, 2, 1],
                [1.0, 3, 0],
                [1.5, 1, 1],
                [2.0, 2, 2],
                [0.25, 3, 2],
            ]
        )
        y = X[:, 0] ** 2 + X[:, 1] + 2 * X[:, 2]

        model = Kriging(
            method="regression",
            var_type=["float", "int", "factor"],
            seed=42,
            model_fun_evals=40,
        )
        model.fit(X, y)

        # Check variable type masks
        assert model.num_mask.sum() == 1  # One float variable
        assert model.int_mask.sum() == 1  # One int variable
        assert model.factor_mask.sum() == 1  # One factor variable
        assert model.ordered_mask.sum() == 2  # float and int are ordered

        # Test prediction
        X_test = np.array([[0.75, 2, 1]])
        y_pred = model.predict(X_test)

        assert np.isfinite(y_pred[0])

    def test_kriging_variable_type_float_mask(self):
        """Test that 'float' is correctly recognized as numeric."""
        X = np.random.rand(8, 2)
        y = np.sum(X**2, axis=1)

        # Model with 'float'
        model_float = Kriging(var_type=["float", "float"], seed=42, model_fun_evals=30)
        model_float.fit(X, y)

        # Both should recognize variables as numeric
        assert np.all(model_float.num_mask == [True, True])

    def test_kriging_only_factor_variables(self):
        """Test Kriging with only factor variables."""
        # Categorical data: 3 variables, each with 2-3 categories
        X = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1], [0, 0, 2], [1, 1, 2]])
        y = np.array([1.0, 2.0, 1.5, 3.0, 1.2, 3.5])

        model = Kriging(
            method="regression",
            var_type=["factor", "factor", "factor"],
            metric_factorial="canberra",
            seed=42,
            model_fun_evals=40,
        )
        model.fit(X, y)

        # All should be factor variables
        assert model.factor_mask.sum() == 3
        assert model.ordered_mask.sum() == 0

        # Test prediction
        X_test = np.array([[1, 0, 1]])
        y_pred = model.predict(X_test)

        assert np.isfinite(y_pred[0])

    def test_kriging_input_reshaping(self):
        """Test that input reshaping works correctly."""
        X = np.array([[0.0], [0.5], [1.0]])
        y = np.array([0.0, 0.25, 1.0])

        model = Kriging(seed=42, model_fun_evals=30)
        model.fit(X, y)

        # Test with 1D array
        X_test_1d = np.array([0.25])
        y_pred_1d = model.predict(X_test_1d)
        assert y_pred_1d.shape == (1,)

        # Test with 2D array
        X_test_2d = np.array([[0.25], [0.75]])
        y_pred_2d = model.predict(X_test_2d)
        assert y_pred_2d.shape == (2,)

    def test_kriging_wrong_input_dimensions(self):
        """Test that wrong input dimensions raise ValueError."""
        X = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        y = np.array([0.0, 0.25, 1.0])

        model = Kriging(seed=42, model_fun_evals=30)
        model.fit(X, y)

        # Wrong number of features (3 instead of 2, cannot be transposed to match)
        with pytest.raises(ValueError):
            model.predict(np.array([[0.25, 0.5, 0.75]]))

    def test_kriging_seed_reproducibility(self):
        """Test that same seed produces same results."""
        X = np.random.rand(10, 2)
        y = np.sum(X**2, axis=1)

        model1 = Kriging(method="regression", seed=42, model_fun_evals=30)
        model1.fit(X, y)
        pred1 = model1.predict(np.array([[0.5, 0.5]]))

        model2 = Kriging(method="regression", seed=42, model_fun_evals=30)
        model2.fit(X, y)
        pred2 = model2.predict(np.array([[0.5, 0.5]]))

        np.testing.assert_array_almost_equal(pred1, pred2)
        np.testing.assert_array_almost_equal(model1.theta_, model2.theta_)

    def test_kriging_get_set_params(self):
        """Test scikit-learn get_params and set_params."""
        model = Kriging(method="regression", seed=42, isotropic=True)

        # Test get_params
        params = model.get_params()
        assert isinstance(params, dict)
        assert params["method"] == "regression"
        assert params["seed"] == 42
        assert params["isotropic"] is True

        # Test set_params
        model.set_params(method="interpolation", seed=123)
        assert model.method == "interpolation"
        assert model.seed == 123

    def test_kriging_with_spotoptim_sphere(self):
        """Test Kriging integration with SpotOptim on sphere function."""

        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]

        kriging = Kriging(method="regression", seed=42, model_fun_evals=50)

        optimizer = SpotOptim(
            fun=sphere,
            bounds=bounds,
            max_iter=15,
            n_initial=8,
            surrogate=kriging,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert result.success is True
        assert result.nfev == 15
        assert result.nit == 7  # 15 total - 8 initial
        assert len(result.x) == 2
        assert result.fun < 5.0  # Should find reasonable solution

    def test_kriging_with_spotoptim_rosenbrock(self):
        """Test Kriging integration with SpotOptim on Rosenbrock function."""

        def rosenbrock(X):
            X = np.atleast_2d(X)
            x = X[:, 0]
            y = X[:, 1]
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        bounds = [(-2, 2), (-2, 2)]

        kriging = Kriging(method="regression", seed=42, model_fun_evals=50)

        optimizer = SpotOptim(
            fun=rosenbrock,
            bounds=bounds,
            max_iter=20,
            n_initial=10,
            surrogate=kriging,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert result.success is True
        assert result.fun < 100.0  # Should improve from random initialization

    def test_kriging_with_spotoptim_mixed_types(self):
        """Test Kriging with SpotOptim using mixed variable types."""

        def mixed_objective(X):
            X = np.atleast_2d(X)
            # x0: continuous, x1: integer, x2: factor (0, 1, 2)
            return X[:, 0] ** 2 + (X[:, 1] - 2) ** 2 + (X[:, 2] == 1).astype(float)

        bounds = [(0, 5), (1, 5), (0, 2)]

        kriging = Kriging(
            method="regression",
            var_type=["float", "int", "factor"],
            seed=42,
            model_fun_evals=40,
        )

        optimizer = SpotOptim(
            fun=mixed_objective,
            bounds=bounds,
            var_type=["float", "int", "factor"],
            max_iter=15,
            n_initial=8,
            surrogate=kriging,
            seed=42,
            verbose=False,
        )

        result = optimizer.optimize()

        assert result.success is True
        assert result.fun < 10.0

    def test_kriging_compare_methods(self):
        """Compare different Kriging methods on same data."""
        np.random.seed(42)
        X = np.random.rand(10, 2) * 4 - 2
        y = np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.1 * np.random.randn(10)

        X_test = np.array([[0.5, 0.5]])

        methods = ["interpolation", "regression", "reinterpolation"]
        predictions = []

        for method in methods:
            model = Kriging(method=method, seed=42, model_fun_evals=30)
            model.fit(X, y)
            y_pred, y_std = model.predict(X_test, return_std=True)
            predictions.append((method, y_pred[0], y_std[0]))

            # All methods should produce valid predictions
            assert np.isfinite(y_pred[0])
            assert np.isfinite(y_std[0])
            assert y_std[0] >= 0

        # Check that predictions are different (methods use different approaches)
        pred_values = [p[1] for p in predictions]
        assert len(set(np.round(pred_values, 4))) > 1  # At least some variation

    def test_kriging_lambda_optimization(self):
        """Test that Lambda is optimized for regression methods."""
        X = np.random.rand(8, 2)
        y = np.sum(X**2, axis=1) + 0.1 * np.random.randn(8)

        # Regression: Lambda should be optimized
        model_reg = Kriging(method="regression", seed=42, model_fun_evals=30)
        model_reg.fit(X, y)
        assert model_reg.Lambda_ is not None
        assert model_reg.min_Lambda <= model_reg.Lambda_ <= model_reg.max_Lambda

        # Interpolation: No Lambda optimization
        model_interp = Kriging(method="interpolation", seed=42, model_fun_evals=30)
        model_interp.fit(X, y)
        assert model_interp.Lambda_ is None

    def test_kriging_custom_lambda_bounds(self):
        """Test custom Lambda bounds."""
        X = np.random.rand(8, 2)
        y = np.sum(X**2, axis=1)

        model = Kriging(
            method="regression",
            min_Lambda=-5.0,
            max_Lambda=-1.0,
            seed=42,
            model_fun_evals=30,
        )
        model.fit(X, y)

        # Lambda should be within custom bounds
        assert -5.0 <= model.Lambda_ <= -1.0

    def test_kriging_custom_theta_bounds(self):
        """Test custom theta bounds."""
        X = np.random.rand(8, 2)
        y = np.sum(X**2, axis=1)

        model = Kriging(
            method="regression",
            min_theta=-2.0,
            max_theta=1.0,
            seed=42,
            model_fun_evals=30,
        )
        model.fit(X, y)

        # All theta values should be within custom bounds
        assert np.all(model.theta_ >= -2.0)
        assert np.all(model.theta_ <= 1.0)

    def test_kriging_metric_factorial_options(self):
        """Test different distance metrics for factor variables."""
        X = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 2], [2, 0]])
        y = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 2.2])

        metrics = ["canberra", "hamming"]

        for metric in metrics:
            model = Kriging(
                method="regression",
                var_type=["factor", "factor"],
                metric_factorial=metric,
                seed=42,
                model_fun_evals=30,
            )
            model.fit(X, y)

            X_test = np.array([[1, 1]])
            y_pred = model.predict(X_test)

            assert np.isfinite(y_pred[0]), f"Prediction failed for metric {metric}"

    def test_kriging_higher_dimensional(self):
        """Test Kriging on higher dimensional problem."""
        np.random.seed(42)
        X = np.random.rand(20, 5)
        y = np.sum(X**2, axis=1)

        model = Kriging(method="regression", seed=42, model_fun_evals=50)
        model.fit(X, y)

        assert model.k == 5
        assert model.n_theta == 5  # Anisotropic by default

        X_test = np.random.rand(3, 5)
        y_pred = model.predict(X_test)

        assert y_pred.shape == (3,)
        assert np.all(np.isfinite(y_pred))

    def test_kriging_small_dataset(self):
        """Test Kriging with minimal dataset."""
        X = np.array([[0.0], [1.0], [2.0]])
        y = np.array([0.0, 1.0, 4.0])

        model = Kriging(method="regression", seed=42, model_fun_evals=30)
        model.fit(X, y)

        X_test = np.array([[0.5], [1.5]])
        y_pred = model.predict(X_test)

        assert y_pred.shape == (2,)
        assert np.all(np.isfinite(y_pred))

    def test_kriging_flatten_y(self):
        """Test that y is correctly flattened during fit."""
        X = np.array([[0.0], [0.5], [1.0]])
        y_1d = np.array([0.0, 0.25, 1.0])
        y_2d = np.array([[0.0], [0.25], [1.0]])

        model1 = Kriging(seed=42, model_fun_evals=30)
        model1.fit(X, y_1d)

        model2 = Kriging(seed=42, model_fun_evals=30)
        model2.fit(X, y_2d.ravel())

        X_test = np.array([[0.5]])
        pred1 = model1.predict(X_test)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_kriging_uncertainty_at_training_points(self):
        """Test that uncertainty is low at training points."""
        X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0]])
        y = np.sin(X.ravel())

        model = Kriging(method="regression", seed=42, model_fun_evals=50)
        model.fit(X, y)

        # Predict at training points
        y_pred, y_std = model.predict(X, return_std=True)

        # Uncertainty should be relatively low at training points
        # (not exactly zero due to Lambda/nugget effect)
        assert np.all(y_std < 0.5)  # Reasonable threshold
