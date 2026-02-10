# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for plot_feature_importances in spotoptim.sensitivity.importance
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.datasets import make_regression

from spotoptim.inspection.importance import plot_feature_importances


@pytest.fixture
def regression_data_numpy():
    X, y = make_regression(n_samples=120, n_features=6, noise=0.2, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    target_names = ["y"]
    return X, y, feature_names, target_names


@pytest.fixture
def regression_data_pandas():
    X, y = make_regression(n_samples=120, n_features=5, noise=0.2, random_state=0)
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="y")
    feature_names = list(X_df.columns)
    target_names = ["y"]
    return X_df.values, y_series.values, feature_names, target_names


class TestPlotFeatureImportancesBasic:
    @patch('matplotlib.pyplot.show')
    def test_returns_top_features_and_df(self, mock_show, regression_data_numpy):
        X, y, feature_names, target_names = regression_data_numpy
        top_features, imp_df = plot_feature_importances(
            X, y, feature_names, target_names, target_index=0, n_top_features=3, figsize=(5, 4)
        )
        # Validate return types
        assert isinstance(top_features, list)
        assert isinstance(imp_df, pd.DataFrame)
        # Validate length of top features
        assert len(top_features) == 3
        # Validate dataframe structure
        assert list(imp_df.columns) == ["feature", "importance"]
        assert imp_df.shape[0] == len(feature_names)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_handles_pandas_like_targets(self, mock_show, regression_data_pandas):
        X, y, feature_names, target_names = regression_data_pandas
        top_features, imp_df = plot_feature_importances(
            X, y, feature_names, target_names, target_index=0, n_top_features=5
        )
        assert len(top_features) == 5
        assert imp_df.shape[0] == len(feature_names)
        mock_show.assert_called_once()


class TestPlotFeatureImportancesEdgeCases:
    @patch('matplotlib.pyplot.show')
    def test_large_n_top_features_capped_by_features(self, mock_show, regression_data_numpy):
        X, y, feature_names, target_names = regression_data_numpy
        n_top = len(feature_names) + 10
        top_features, imp_df = plot_feature_importances(
            X, y, feature_names, target_names, target_index=0, n_top_features=n_top
        )
        # Should not exceed number of features
        assert len(top_features) <= len(feature_names)
        assert imp_df.shape[0] == len(feature_names)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_multiple_targets_select_by_index(self, mock_show, regression_data_numpy):
        X, y, feature_names, target_names = regression_data_numpy
        # Create two target names; still using single y but index should select name safely
        target_names = ["t0", "t1"]
        top_features, imp_df = plot_feature_importances(
            X, y, feature_names, target_names, target_index=1, n_top_features=4
        )
        assert isinstance(top_features, list)
        assert imp_df.shape[0] == len(feature_names)
        mock_show.assert_called_once()


class TestPlotFeatureImportancesIntegration:
    @patch('matplotlib.pyplot.show')
    def test_full_parameter_combo(self, mock_show, regression_data_numpy):
        X, y, feature_names, target_names = regression_data_numpy
        top_features, imp_df = plot_feature_importances(
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            target_index=0,
            n_top_features=6,
            figsize=(8, 6),
        )
        assert len(top_features) == 6
        assert set(top_features).issubset(set(feature_names))
        assert imp_df.shape[0] == len(feature_names)
        mock_show.assert_called_once()
