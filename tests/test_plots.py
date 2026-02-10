# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for spotoptim.eda.plots module.

This module contains comprehensive tests for the plotting functions:
- plot_ip_histograms: Tests for infill-point histogram generation
- plot_ip_boxplots: Tests for infill-point boxplot generation
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from spotoptim.eda.plots import plot_ip_histograms, plot_ip_boxplots


@pytest.fixture
def sample_df():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'C': [5, 5, 5, 5, 5, 5, 5, 5, 5, 5]  # Low unique values
    })


@pytest.fixture
def sample_df_with_outliers():
    """Create a DataFrame with outliers."""
    return pd.DataFrame({
        'A': [1, 2, 2, 3, 4, 5, 100],  # 100 is an outlier
        'B': [10, 10, 10, 10, 10, 10, 10]
    })


@pytest.fixture
def sample_df_with_nans():
    """Create a DataFrame with NaN values."""
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50]
    })


@pytest.fixture
def sample_df_with_categories():
    """Create a DataFrame with categorical column."""
    return pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 6, 7, 8],
        'category': ['X', 'X', 'Y', 'Y', 'X', 'X', 'Y', 'Y']
    })


@pytest.fixture
def additional_points_df():
    """Create additional points DataFrame for highlighting."""
    return pd.DataFrame({
        'A': [2.5, 7.5],
        'B': [25, 75]
    })


# Tests for plot_ip_histograms

class TestPlotIpHistogramsBasic:
    """Test basic functionality of plot_ip_histograms."""

    @patch('matplotlib.pyplot.show')
    def test_basic_plotting(self, mock_show, sample_df):
        """Test basic histogram plotting."""
        plot_ip_histograms(sample_df)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_custom_bins(self, mock_show, sample_df):
        """Test histogram with custom number of bins."""
        plot_ip_histograms(sample_df, bins=20)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_single_column_layout(self, mock_show, sample_df):
        """Test histogram with single column layout."""
        plot_ip_histograms(sample_df, num_cols=1)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_three_column_layout(self, mock_show, sample_df):
        """Test histogram with three column layout."""
        plot_ip_histograms(sample_df, num_cols=3)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_custom_figure_width(self, mock_show, sample_df):
        """Test histogram with custom figure width."""
        plot_ip_histograms(sample_df, figwidth=15)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpHistogramsThreshold:
    """Test threshold-based coloring in plot_ip_histograms."""

    @patch('matplotlib.pyplot.show')
    def test_with_low_unique_threshold(self, mock_show, sample_df):
        """Test histogram with low unique value threshold."""
        plot_ip_histograms(sample_df, thrs_unique=3)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_high_unique_threshold(self, mock_show, sample_df):
        """Test histogram with high unique value threshold."""
        plot_ip_histograms(sample_df, thrs_unique=15)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_color_changes_with_threshold(self, mock_show, sample_df):
        """Test that histogram color changes based on unique value threshold."""
        # Column C has only 1 unique value, should trigger lightcoral color
        plot_ip_histograms(sample_df, thrs_unique=5)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpHistogramsAdditionalPoints:
    """Test additional points functionality in plot_ip_histograms."""

    @patch('matplotlib.pyplot.show')
    def test_with_additional_points(self, mock_show, sample_df, additional_points_df):
        """Test histogram with additional points highlighted."""
        plot_ip_histograms(sample_df, add_points=additional_points_df, add_points_col=["red", "blue"])
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_partial_additional_points(self, mock_show, sample_df):
        """Test histogram with additional points for only some columns."""
        partial_points = pd.DataFrame({'A': [5, 6]})
        plot_ip_histograms(sample_df, add_points=partial_points, add_points_col=["red", "blue"])
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_additional_points_with_nans(self, mock_show, sample_df):
        """Test histogram with additional points containing NaN values."""
        points_with_nan = pd.DataFrame({'A': [5, np.nan], 'B': [50, 60]})
        plot_ip_histograms(sample_df, add_points=points_with_nan, add_points_col=["red", "blue"])
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpHistogramsEdgeCases:
    """Test edge cases for plot_ip_histograms."""

    @patch('matplotlib.pyplot.show')
    def test_with_single_column(self, mock_show):
        """Test histogram with DataFrame containing single column."""
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        plot_ip_histograms(df)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_many_columns(self, mock_show):
        """Test histogram with DataFrame containing many columns."""
        df = pd.DataFrame({f'col_{i}': np.random.randn(10) for i in range(10)})
        plot_ip_histograms(df, num_cols=3)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_nans(self, mock_show, sample_df_with_nans):
        """Test histogram with NaN values (should be dropped)."""
        plot_ip_histograms(sample_df_with_nans)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_outliers(self, mock_show, sample_df_with_outliers):
        """Test histogram with outliers."""
        plot_ip_histograms(sample_df_with_outliers)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_constant_values(self, mock_show):
        """Test histogram with constant values."""
        df = pd.DataFrame({'A': [5] * 10, 'B': [10] * 10})
        plot_ip_histograms(df)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_mixed_dtypes(self, mock_show):
        """Test histogram with mixed data types (should plot only numeric)."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'string': ['a', 'b', 'c', 'd', 'e'],
            'numeric2': [10, 20, 30, 40, 50]
        })
        plot_ip_histograms(df)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpHistogramsReturnType:
    """Test return type of plot_ip_histograms."""

    @patch('matplotlib.pyplot.show')
    def test_returns_none(self, mock_show, sample_df):
        """Test that function returns None."""
        result = plot_ip_histograms(sample_df)
        assert result is None
        plt.close('all')


# Tests for plot_ip_boxplots

class TestPlotIpBoxplotsBasic:
    """Test basic functionality of plot_ip_boxplots."""

    @patch('matplotlib.pyplot.show')
    def test_basic_plotting(self, mock_show, sample_df):
        """Test basic boxplot plotting."""
        plot_ip_boxplots(sample_df)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_single_column_layout(self, mock_show, sample_df):
        """Test boxplot with single column layout."""
        plot_ip_boxplots(sample_df, num_cols=1)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_three_column_layout(self, mock_show, sample_df):
        """Test boxplot with three column layout."""
        plot_ip_boxplots(sample_df, num_cols=3)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_custom_figure_width(self, mock_show, sample_df):
        """Test boxplot with custom figure width."""
        plot_ip_boxplots(sample_df, figwidth=15)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_custom_box_width(self, mock_show, sample_df):
        """Test boxplot with custom box width."""
        plot_ip_boxplots(sample_df, box_width=0.5)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_custom_height_per_subplot(self, mock_show, sample_df):
        """Test boxplot with custom height per subplot."""
        plot_ip_boxplots(sample_df, height_per_subplot=3.0)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpBoxplotsCategories:
    """Test category-based grouping in plot_ip_boxplots."""

    @patch('matplotlib.pyplot.show')
    def test_with_categories(self, mock_show, sample_df_with_categories):
        """Test boxplot with categorical grouping."""
        plot_ip_boxplots(sample_df_with_categories, category_column_name='category')
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_without_categories(self, mock_show, sample_df):
        """Test boxplot without categorical grouping."""
        plot_ip_boxplots(sample_df, category_column_name=None)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_invalid_category_column(self, mock_show, sample_df):
        """Test boxplot with non-existent category column."""
        plot_ip_boxplots(sample_df, category_column_name='nonexistent')
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_both_names_true(self, mock_show, sample_df_with_categories):
        """Test boxplot with both_names=True."""
        plot_ip_boxplots(sample_df_with_categories, both_names=True)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_both_names_false(self, mock_show, sample_df_with_categories):
        """Test boxplot with both_names=False."""
        plot_ip_boxplots(sample_df_with_categories, both_names=False)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpBoxplotsAdditionalPoints:
    """Test additional points functionality in plot_ip_boxplots."""

    @patch('matplotlib.pyplot.show')
    def test_with_additional_points(self, mock_show, sample_df, additional_points_df):
        """Test boxplot with additional points highlighted."""
        plot_ip_boxplots(sample_df, add_points=additional_points_df, add_points_col=["red", "blue"])
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_partial_additional_points(self, mock_show, sample_df):
        """Test boxplot with additional points for only some columns."""
        partial_points = pd.DataFrame({'A': [5, 6]})
        plot_ip_boxplots(sample_df, add_points=partial_points, add_points_col=["red", "blue"])
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_additional_points_with_nans(self, mock_show, sample_df):
        """Test boxplot with additional points containing NaN values."""
        points_with_nan = pd.DataFrame({'A': [5, np.nan], 'B': [50, 60]})
        plot_ip_boxplots(sample_df, add_points=points_with_nan, add_points_col=["red", "blue"])
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpBoxplotsEdgeCases:
    """Test edge cases for plot_ip_boxplots."""

    @patch('matplotlib.pyplot.show')
    def test_with_series_input(self, mock_show):
        """Test boxplot with Series input (should convert to DataFrame)."""
        series = pd.Series([1, 2, 3, 4, 5], name='A')
        plot_ip_boxplots(series)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_single_column(self, mock_show):
        """Test boxplot with DataFrame containing single column."""
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        plot_ip_boxplots(df)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_many_columns(self, mock_show):
        """Test boxplot with DataFrame containing many columns."""
        df = pd.DataFrame({f'col_{i}': np.random.randn(10) for i in range(10)})
        plot_ip_boxplots(df, num_cols=3)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_nans(self, mock_show, sample_df_with_nans):
        """Test boxplot with NaN values (should be dropped)."""
        plot_ip_boxplots(sample_df_with_nans)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_outliers(self, mock_show, sample_df_with_outliers):
        """Test boxplot with outliers."""
        plot_ip_boxplots(sample_df_with_outliers)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_constant_values(self, mock_show):
        """Test boxplot with constant values."""
        df = pd.DataFrame({'A': [5] * 10, 'B': [10] * 10})
        plot_ip_boxplots(df)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_mixed_dtypes(self, mock_show):
        """Test boxplot with mixed data types (should plot only numeric)."""
        df = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'string': ['a', 'b', 'c', 'd', 'e'],
            'numeric2': [10, 20, 30, 40, 50]
        })
        plot_ip_boxplots(df)
        mock_show.assert_called_once()
        plt.close('all')

    @patch('matplotlib.pyplot.show')
    def test_with_small_dataset(self, mock_show):
        """Test boxplot with very small dataset."""
        df = pd.DataFrame({'A': [1, 2], 'B': [10, 20]})
        plot_ip_boxplots(df)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpBoxplotsReturnType:
    """Test return type of plot_ip_boxplots."""

    @patch('matplotlib.pyplot.show')
    def test_returns_none(self, mock_show, sample_df):
        """Test that function returns None."""
        result = plot_ip_boxplots(sample_df)
        assert result is None
        plt.close('all')


class TestPlotIpHistogramsDocstring:
    """Test docstring examples for plot_ip_histograms."""

    @patch('matplotlib.pyplot.show')
    def test_docstring_example(self, mock_show):
        """Test the example from the docstring."""
        data = {'A': [1, 2, 2, 3, 4, 5, 100], 'B': [10, 10, 10, 10, 10, 10, 10]}
        df = pd.DataFrame(data)
        plot_ip_histograms(df, bins=5, num_cols=1, thrs_unique=3)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpBoxplotsDocstring:
    """Test docstring examples for plot_ip_boxplots."""

    @patch('matplotlib.pyplot.show')
    def test_docstring_example(self, mock_show):
        """Test the example from the docstring."""
        data = {'A': [1, 2, 2, 3, 4, 5, 100], 'B': [10, 10, 10, 10, 10, 10, 10]}
        df = pd.DataFrame(data)
        plot_ip_boxplots(df, num_cols=1)
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpHistogramsIntegration:
    """Integration tests for plot_ip_histograms."""

    @patch('matplotlib.pyplot.show')
    def test_full_parameter_combination(self, mock_show, sample_df, additional_points_df):
        """Test histogram with all parameters combined."""
        plot_ip_histograms(
            df=sample_df,
            bins=15,
            num_cols=2,
            figwidth=12,
            thrs_unique=7,
            add_points=additional_points_df,
            add_points_col=["red", "blue"]
        )
        mock_show.assert_called_once()
        plt.close('all')


class TestPlotIpBoxplotsIntegration:
    """Integration tests for plot_ip_boxplots."""

    @patch('matplotlib.pyplot.show')
    def test_full_parameter_combination(self, mock_show, sample_df_with_categories, additional_points_df):
        """Test boxplot with all parameters combined."""
        # Add 'value' column to additional_points_df
        add_pts = pd.DataFrame({'value': [2.5, 6.5]})
        plot_ip_boxplots(
            df=sample_df_with_categories,
            category_column_name='category',
            num_cols=1,
            figwidth=12,
            box_width=0.3,
            both_names=True,
            height_per_subplot=3.0,
            add_points=add_pts,
            add_points_col=["red", "blue"]
        )
        mock_show.assert_called_once()
        plt.close('all')
