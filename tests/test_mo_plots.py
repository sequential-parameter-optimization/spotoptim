# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from spotoptim.mo.pareto import mo_xy_surface, mo_xy_contour


# Mock model class that mimics sklearn estimator
class MockModel:
    def __init__(self, n_features_in_=2):
        self.n_features_in_ = n_features_in_

    def predict(self, X):
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but MockModel is expecting {self.n_features_in_} features as input."
            )
        # Return sum of squares as dummy prediction
        return np.sum(X**2, axis=1)


@pytest.fixture
def mock_plt():
    with (
        patch("matplotlib.pyplot.figure") as mock_fig,
        patch("matplotlib.pyplot.subplots") as mock_subplots,
        patch("matplotlib.pyplot.show"),
    ):

        # Setup mock figure and axes
        fig = MagicMock()
        ax = MagicMock()
        mock_fig.return_value = fig
        # subplots returns (fig, ax) or (fig, axes_array)
        # We return a 2x2 array to cover up to 4 plots, which is enough for our tests
        # Use dtype=object and fill manually to avoid numpy treating mocks as sequences
        axes_array = np.empty((2, 2), dtype=object)
        axes_array.fill(ax)
        mock_subplots.return_value = (fig, axes_array)
        fig.add_subplot.return_value = ax

        yield mock_subplots


def test_mo_xy_surface_valid(mock_plt):
    models = [MockModel(2), MockModel(2)]
    bounds = [(0, 1), (0, 1)]

    # Should run without error
    mo_xy_surface(models, bounds)


def test_mo_xy_contour_valid(mock_plt):
    models = [MockModel(2), MockModel(2)]
    bounds = [(0, 1), (0, 1)]

    # Should run without error
    mo_xy_contour(models, bounds)


def test_mo_xy_surface_invalid_bounds_scalar():
    models = [MockModel(2)]
    # Invalid bounds (arrays instead of scalars)
    bounds = [(np.array([0, 0]), np.array([1, 1])), (0, 1)]

    with pytest.raises(ValueError, match="must be scalars"):
        mo_xy_surface(models, bounds)


def test_mo_xy_surface_mismatch_features():
    # Model expects 2 features, but we provide 3 bounds
    models = [MockModel(2)]
    bounds = [(0, 1), (0, 1), (0, 1)]

    with pytest.raises(
        ValueError, match="expects 2 features, but 3 bounds were provided"
    ):
        mo_xy_surface(models, bounds)


def test_mo_xy_surface_invalid_feature_pair():
    models = [MockModel(3)]
    bounds = [(0, 1), (0, 1), (0, 1)]

    # Index 5 is out of bounds for 3 features
    feature_pairs = [(0, 5)]

    with pytest.raises(ValueError, match="Invalid feature pair"):
        mo_xy_surface(models, bounds, feature_pairs=feature_pairs)


def test_mo_xy_contour_feature_subset(mock_plt):
    models = [MockModel(3)]
    bounds = [(0, 1), (0, 1), (0, 1)]

    # Only plot pair (0, 2)
    mo_xy_contour(models, bounds, feature_pairs=[(0, 2)])
    # We can't easily assert exactly what was plotted without complex mock inspection,
    # but verifying it runs without error is the primary goal here.


def test_model_predict_error_handling():
    # Model that raises ValueError on predict (simulating sklearn check)
    model = MagicMock()
    model.n_features_in_ = 2
    model.predict.side_effect = ValueError(
        "X has 3 features, but is expecting 2 features"
    )

    # We pass 3 bounds to a 2-feature model (if we bypass the explicit check, or if n_features_in_ is missing)
    # Let's simulate a model WITHOUT n_features_in_ to test the try-except in _get_mo_plot_data
    del model.n_features_in_

    bounds = [(0, 1), (0, 1), (0, 1)]

    # mo_xy_surface will assume 3 features based on bounds
    # _get_mo_plot_data will build X_grid with 3 columns
    # model.predict will raise ValueError
    # We expect our caught exception re-raise

    with pytest.raises(
        ValueError, match=r"len\(bounds\) matches the model's input dimensionality"
    ):
        mo_xy_surface([model], bounds)
