# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for importance utilities in spotoptim.sensitivity.importance:
- generate_mdi
- generate_imp
- plot_importances
"""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sklearn.datasets import make_regression

from spotoptim.inspection.importance import generate_mdi, generate_imp, plot_importances


@pytest.fixture
def regression_splits():
    X, y = make_regression(n_samples=120, n_features=6, noise=0.2, random_state=42)
    # Return both numpy and pandas variants
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y_series = pd.Series(y, name="y")
    return X, y, X_df, y_series


class TestGenerateMDI:
    def test_returns_dataframe_with_expected_columns_numpy(self, regression_splits):
        X, y, _, _ = regression_splits
        df_mdi = generate_mdi(X, y)
        assert isinstance(df_mdi, pd.DataFrame)
        assert list(df_mdi.columns) == ["Feature", "Importance"]
        assert df_mdi.shape[0] == X.shape[1]
        # Importance values should be non-negative
        assert (df_mdi["Importance"] >= 0).all()

    def test_returns_dataframe_with_feature_names_pandas(self, regression_splits):
        _, _, X_df, y_series = regression_splits
        feature_names = list(X_df.columns)
        df_mdi = generate_mdi(X_df, y_series, feature_names=feature_names)
        # Order is sorted by importance; verify contents and length
        assert set(df_mdi["Feature"].tolist()) == set(feature_names)
        assert df_mdi.shape[0] == X_df.shape[1]

    def test_handles_y_as_array(self, regression_splits):
        X, y, _, _ = regression_splits
        df_mdi = generate_mdi(X, y)
        assert isinstance(df_mdi, pd.DataFrame)

    def test_raises_on_length_mismatch(self, regression_splits):
        X, y, _, _ = regression_splits
        # Make y shorter to trigger an error in sklearn fit
        y_bad = y[:-1]
        with pytest.raises((ValueError, AssertionError)):
            generate_mdi(X, y_bad)


class TestGenerateIMP:
    def test_returns_perm_importance_object_numpy(self, regression_splits):
        X, y, _, _ = regression_splits
        # Create simple train/test split
        X_train, X_test = X[:90], X[90:]
        y_train, y_test = y[:90], y[90:]
        perm_imp = generate_imp(X_train, X_test, y_train, y_test, n_repeats=5, use_test=True)
        # Check structure
        assert hasattr(perm_imp, "importances")
        assert hasattr(perm_imp, "importances_mean")
        # importances_mean length should match number of features
        assert perm_imp.importances_mean.shape[0] == X_train.shape[1]

    def test_handles_pandas_inputs(self, regression_splits):
        _, _, X_df, y_series = regression_splits
        X_train, X_test = X_df.iloc[:90], X_df.iloc[90:]
        y_train, y_test = y_series.iloc[:90], y_series.iloc[90:]
        perm_imp = generate_imp(X_train, X_test, y_train, y_test, n_repeats=3, use_test=False)
        assert hasattr(perm_imp, "importances_mean")

    def test_use_train_vs_test_flag(self, regression_splits):
        X, y, _, _ = regression_splits
        X_train, X_test = X[:90], X[90:]
        y_train, y_test = y[:90], y[90:]
        perm_test = generate_imp(X_train, X_test, y_train, y_test, n_repeats=3, use_test=True)
        perm_train = generate_imp(X_train, X_test, y_train, y_test, n_repeats=3, use_test=False)
        # Means may differ but shapes should match
        assert perm_test.importances.shape == perm_train.importances.shape

    def test_raises_on_shape_mismatch_between_X_and_y(self, regression_splits):
        X, y, _, _ = regression_splits
        X_train, X_test = X[:90], X[90:]
        y_train = y[:85]  # shorter
        y_test = y[90:]
        with pytest.raises(ValueError):
            generate_imp(X_train, X_test, y_train, y_test)

    def test_raises_on_feature_count_mismatch_train_test(self, regression_splits):
        X, y, _, _ = regression_splits
        X_train, X_test = X[:90], X[90:]
        y_train, y_test = y[:90], y[90:]
        # Remove a column from X_test to mismatch feature counts
        X_test_bad = X_test[:, :-1]
        with pytest.raises(ValueError):
            generate_imp(X_train, X_test_bad, y_train, y_test)


class TestPlotImportances:
    @patch('matplotlib.pyplot.show')
    def test_basic_plot(self, mock_show, regression_splits):
        X, y, X_df, y_series = regression_splits
        # Prepare inputs
        df_mdi = generate_mdi(X_df, y_series)
        X_train, X_test = X_df.iloc[:90], X_df.iloc[90:]
        y_train, y_test = y_series.iloc[:90], y_series.iloc[90:]
        perm_imp = generate_imp(X_train, X_test, y_train, y_test, n_repeats=3)
        # Plot with feature names
        plot_importances(
            df_mdi=df_mdi,
            perm_imp=perm_imp,
            X_test=X_test,
            target_name="y",
            feature_names=list(X_df.columns),
            k=5,
            figsize=(8, 6),
            show=True,
        )
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_raises_on_k_bigger_than_features(self, mock_show, regression_splits):
        X, y, X_df, y_series = regression_splits
        df_mdi = generate_mdi(X_df, y_series)
        X_train, X_test = X_df.iloc[:90], X_df.iloc[90:]
        y_train, y_test = y_series.iloc[:90], y_series.iloc[90:]
        perm_imp = generate_imp(X_train, X_test, y_train, y_test, n_repeats=3)
        # k larger than number of features should still work due to slicing, not raise
        plot_importances(
            df_mdi=df_mdi,
            perm_imp=perm_imp,
            X_test=X_test,
            target_name=None,
            feature_names=None,
            k=X_df.shape[1] + 10,
            figsize=(6, 4),
            show=True,
        )
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_without_feature_names(self, mock_show, regression_splits):
        X, y, X_df, y_series = regression_splits
        df_mdi = generate_mdi(X_df, y_series)
        X_train, X_test = X_df.iloc[:90], X_df.iloc[90:]
        y_train, y_test = y_series.iloc[:90], y_series.iloc[90:]
        perm_imp = generate_imp(X_train, X_test, y_train, y_test, n_repeats=3)
        # Feature names omitted; should fall back to X_test columns
        plot_importances(
            df_mdi=df_mdi,
            perm_imp=perm_imp,
            X_test=X_test,
            target_name=None,
            feature_names=None,
            k=4,
            figsize=(6, 4),
            show=True,
        )
        mock_show.assert_called_once()
