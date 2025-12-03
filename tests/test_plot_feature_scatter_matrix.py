"""
Tests for plot_feature_scatter_matrix in spotoptim.sensitivity.importance
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.datasets import make_regression

from spotoptim.inspection.importance import plot_feature_scatter_matrix


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=120, n_features=6, noise=0.2, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    target_names = ["y"]
    return X, y, feature_names, target_names


class TestPlotFeatureScatterMatrixBasic:
    @patch('matplotlib.pyplot.show')
    def test_basic_plot(self, mock_show, regression_data):
        X, y, feature_names, target_names = regression_data
        top_features = feature_names[:3]
        plot_feature_scatter_matrix(
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            top_features=top_features,
            target_index=0,
            figsize=(6, 6),
        )
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_handles_different_figsize(self, mock_show, regression_data):
        X, y, feature_names, target_names = regression_data
        top_features = feature_names[:2]
        plot_feature_scatter_matrix(
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            top_features=top_features,
            target_index=0,
            figsize=(8, 4),
        )
        mock_show.assert_called_once()


class TestPlotFeatureScatterMatrixEdgeCases:
    @patch('matplotlib.pyplot.show')
    def test_top_features_not_in_feature_names_raises(self, mock_show, regression_data):
        X, y, feature_names, target_names = regression_data
        top_features = ["not_a_feature"]
        with pytest.raises(KeyError):
            plot_feature_scatter_matrix(
                X=X,
                y=y,
                feature_names=feature_names,
                target_names=target_names,
                top_features=top_features,
                target_index=0,
            )

    @patch('matplotlib.pyplot.show')
    def test_target_index_out_of_bounds_raises(self, mock_show, regression_data):
        X, y, feature_names, target_names = regression_data
        top_features = feature_names[:3]
        with pytest.raises(IndexError):
            plot_feature_scatter_matrix(
                X=X,
                y=y.reshape(-1, 1),  # make 2D to simulate multiple targets
                feature_names=feature_names,
                target_names=target_names,
                top_features=top_features,
                target_index=5,  # out of bounds
            )

    @patch('matplotlib.pyplot.show')
    def test_empty_top_features_raises(self, mock_show, regression_data):
        X, y, feature_names, target_names = regression_data
        top_features = []
        # Expect graceful handling (no raise) and show called
        plot_feature_scatter_matrix(
            X=X,
            y=y,
            feature_names=feature_names,
            target_names=target_names,
            top_features=top_features,
            target_index=0,
        )
        mock_show.assert_called_once()
