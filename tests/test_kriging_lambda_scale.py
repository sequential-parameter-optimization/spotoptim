# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the Kriging nugget search parametrization (lambda_scale)."""

import numpy as np
import pytest

from spotoptim.surrogate import Kriging


def _noisy_data(seed=0, n=30):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 2))
    y = np.sum(X**2, axis=1) + rng.normal(0, 0.5, size=n)
    return X, y


class TestKrigingLambdaScale:
    """Test suite for the lambda_scale option."""

    def test_default_is_log10(self):
        model = Kriging()
        assert model.lambda_scale == "log10"

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="lambda_scale"):
            Kriging(lambda_scale="quadratic")

    def test_log10_lambda_within_exponent_bounds(self):
        X, y = _noisy_data()
        model = Kriging(method="regression", seed=42).fit(X, y)
        assert model.min_Lambda <= model.Lambda_ <= model.max_Lambda

    def test_linear_lambda_within_value_bounds(self):
        X, y = _noisy_data()
        model = Kriging(method="regression", lambda_scale="linear", seed=42).fit(X, y)
        assert 10.0**model.min_Lambda <= model.Lambda_ <= 10.0**model.max_Lambda

    def test_linear_predictions_finite(self):
        X, y = _noisy_data()
        model = Kriging(method="regression", lambda_scale="linear", seed=42).fit(X, y)
        pred, std = model.predict(X[:5], return_std=True)
        assert np.all(np.isfinite(pred))
        assert np.all(np.isfinite(std))
        assert np.all(std >= 0)

    def test_scales_search_identical_nugget_range(self):
        """Both scales cover the same Lambda value range [1e-9, 1]."""
        log_model = Kriging(method="regression", seed=42)
        lin_model = Kriging(method="regression", lambda_scale="linear", seed=42)
        assert log_model._lambda_value(log_model._lambda_bounds()[0]) == pytest.approx(
            lin_model._lambda_value(lin_model._lambda_bounds()[0])
        )
        assert log_model._lambda_value(log_model._lambda_bounds()[1]) == pytest.approx(
            lin_model._lambda_value(lin_model._lambda_bounds()[1])
        )

    def test_likelihood_agrees_across_scales_for_same_nugget(self):
        """The likelihood of a given (theta, Lambda) pair is scale-invariant."""
        X, y = _noisy_data()
        log_model = Kriging(method="regression", seed=42).fit(X, y)
        lin_model = Kriging(method="regression", lambda_scale="linear", seed=42).fit(
            X, y
        )
        theta = log_model.theta_.copy()
        lambda_value = 1e-3
        nll_log, _, _ = log_model.likelihood(
            np.concatenate([theta, [np.log10(lambda_value)]])
        )
        nll_lin, _, _ = lin_model.likelihood(np.concatenate([theta, [lambda_value]]))
        assert nll_log == pytest.approx(nll_lin)

    def test_interpolation_ignores_lambda_scale(self):
        X, y = _noisy_data()
        model = Kriging(method="interpolation", lambda_scale="linear", seed=42).fit(
            X, y
        )
        assert model.Lambda_ is None
        assert np.all(np.isfinite(model.predict(X[:5])))
