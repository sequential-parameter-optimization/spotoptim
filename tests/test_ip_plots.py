# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comprehensive pytest suite for spotoptim.eda.plots.

Covers _align_add_points, plot_ip_histograms, and plot_ip_boxplots.
Run with: uv run pytest tests/test_ip_plots.py -v
"""

import unittest.mock

import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from spotoptim.eda.plots import (
    _align_add_points,
    plot_ip_histograms,
    plot_ip_boxplots,
)

# ---------------------------------------------------------------------------
# Fixture: suppress plt.show() and close figures after every test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def no_show_close(monkeypatch):
    """Suppress plt.show() and close all figures after each test."""
    monkeypatch.setattr(plt, "show", lambda: None)
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Helpers re-used across tests
# ---------------------------------------------------------------------------


def make_df():
    """Return a small reference DataFrame with two numeric columns."""
    return pd.DataFrame(
        {"A": [1, 2, 2, 3, 4, 5, 100], "B": [10, 10, 10, 10, 10, 10, 10]}
    )


def make_add_points_same_names():
    """Return add_points whose column names match the reference DataFrame."""
    return pd.DataFrame({"A": [1.5, 3.5], "B": [10.0, 10.0]})


def make_add_points_diff_names():
    """Return add_points whose column names differ from the reference DataFrame."""
    return pd.DataFrame({"x": [1.5, 3.5], "y": [10.0, 10.0]})


def make_add_points_one_row():
    """Return add_points with a single row."""
    return pd.DataFrame({"A": [2.0], "B": [10.0]})


# ===========================================================================
# _align_add_points
# ===========================================================================


class TestAlignAddPoints:
    """Tests for the _align_add_points helper function."""

    def test_same_names_unchanged(self):
        """add_points whose column names already match df are returned unchanged."""
        df = make_df()
        ap = make_add_points_same_names()
        result = _align_add_points(df, ap)
        assert result.columns.tolist() == ["A", "B"]

    def test_different_names_renamed(self):
        """add_points with different names but same column count are renamed."""
        df = make_df()
        ap = make_add_points_diff_names()
        result = _align_add_points(df, ap)
        assert result.columns.tolist() == ["A", "B"]

    def test_values_preserved_after_rename(self):
        """Numeric values must not change when columns are renamed."""
        df = make_df()
        ap = make_add_points_diff_names()
        result = _align_add_points(df, ap)
        np.testing.assert_array_equal(result["A"].values, ap["x"].values)
        np.testing.assert_array_equal(result["B"].values, ap["y"].values)

    def test_mismatched_column_count_returned_as_is(self):
        """add_points with a different number of numeric columns are not renamed."""
        df = make_df()  # 2 numeric columns
        ap = pd.DataFrame({"x": [1.5], "y": [2.5], "z": [3.5]})  # 3 numeric columns
        result = _align_add_points(df, ap)
        assert result.columns.tolist() == ["x", "y", "z"]

    def test_non_numeric_columns_in_add_points_preserved(self):
        """Non-numeric columns in add_points survive the rename operation."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        ap = pd.DataFrame({"x": [1.5], "y": [2.5], "label": ["point"]})
        # 2 numeric cols in df, 2 numeric cols in ap -> rename x->A, y->B
        result = _align_add_points(df, ap)
        assert "label" in result.columns
        assert result["label"].tolist() == ["point"]

    def test_returns_copy(self):
        """The returned DataFrame must be a copy, not a view of the input."""
        df = make_df()
        ap = make_add_points_diff_names()
        result = _align_add_points(df, ap)
        result["A"] = 999
        assert ap["x"].tolist() != [999, 999]

    def test_type_error_df_not_dataframe(self):
        """TypeError is raised when df is not a pandas DataFrame."""
        with pytest.raises(TypeError):
            _align_add_points([1, 2, 3], pd.DataFrame({"a": [1]}))

    def test_type_error_add_points_not_dataframe(self):
        """TypeError is raised when add_points is not a pandas DataFrame."""
        df = make_df()
        with pytest.raises(TypeError):
            _align_add_points(df, [[1, 2], [3, 4]])

    def test_single_column_rename(self):
        """Single-column DataFrames are renamed correctly."""
        df = pd.DataFrame({"target": [1, 2, 3]})
        ap = pd.DataFrame({"src": [1.5]})
        result = _align_add_points(df, ap)
        assert result.columns.tolist() == ["target"]

    def test_empty_add_points(self):
        """Empty add_points DataFrame with matching column count is renamed."""
        df = pd.DataFrame({"A": [1], "B": [2]})
        ap = pd.DataFrame(
            {"x": pd.Series([], dtype=float), "y": pd.Series([], dtype=float)}
        )
        result = _align_add_points(df, ap)
        assert result.columns.tolist() == ["A", "B"]
        assert len(result) == 0


# ===========================================================================
# plot_ip_histograms
# ===========================================================================


class TestPlotIpHistograms:
    """Tests for plot_ip_histograms."""

    # --- basic smoke tests --------------------------------------------------

    def test_basic_no_add_points(self):
        """Function completes without error for a simple DataFrame."""
        df = make_df()
        plot_ip_histograms(df)

    def test_single_column(self):
        """Function handles a single-column DataFrame."""
        df = pd.DataFrame({"X": [1, 2, 3, 4, 5]})
        plot_ip_histograms(df, num_cols=1)

    def test_many_columns_grid(self):
        """Grid layout is constructed for more than num_cols columns."""
        df = pd.DataFrame(
            {"A": range(10), "B": range(10), "C": range(10), "D": range(10)}
        )
        plot_ip_histograms(df, num_cols=2)

    def test_custom_bins(self):
        """Custom bin count does not raise."""
        df = make_df()
        plot_ip_histograms(df, bins=20)

    def test_thrs_unique_changes_color(self):
        """Low-unique-value columns trigger lightcoral (no assertion on color,
        but no exception should be raised)."""
        df = pd.DataFrame({"constant": [1, 1, 1, 1, 1]})
        plot_ip_histograms(df, thrs_unique=3)

    # --- add_points with matching names ------------------------------------

    def test_add_points_same_names(self):
        """add_points with matching column names are overlaid without error."""
        df = make_df()
        ap = make_add_points_same_names()
        plot_ip_histograms(df, add_points=ap, add_points_col=["red", "blue"])

    def test_add_points_single_point(self):
        """A single-row add_points is accepted."""
        df = make_df()
        ap = make_add_points_one_row()
        plot_ip_histograms(df, add_points=ap, add_points_col=["green"])

    # --- add_points with different names -----------------------------------

    def test_add_points_different_names_aligned(self):
        """add_points with different column names are silently aligned."""
        df = make_df()
        ap = make_add_points_diff_names()
        # Must not raise even though column names differ
        plot_ip_histograms(df, add_points=ap, add_points_col=["red", "blue"])

    def test_add_points_partial_column_overlap(self):
        """add_points that only share some column names still plot the matching ones."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        # add_points has 3 numeric cols -> aligned to A, B, C
        ap = pd.DataFrame({"p": [1.5], "q": [4.5], "r": [7.5]})
        plot_ip_histograms(df, add_points=ap, add_points_col=["purple"])

    # --- error cases --------------------------------------------------------

    def test_add_points_col_length_mismatch_raises(self):
        """ValueError is raised when add_points rows != add_points_col length."""
        df = make_df()
        ap = make_add_points_same_names()  # 2 rows
        with pytest.raises(ValueError, match="Length of add_points"):
            plot_ip_histograms(df, add_points=ap, add_points_col=["red"])  # 1 colour

    # --- NaN handling -------------------------------------------------------

    def test_add_points_with_nan_skipped(self):
        """NaN entries in add_points are dropped before scatter-plotting."""
        df = make_df()
        ap = pd.DataFrame({"A": [1.5, np.nan], "B": [10.0, np.nan]})
        plot_ip_histograms(df, add_points=ap, add_points_col=["red", "blue"])

    def test_df_with_nan_values(self):
        """NaN values in df do not prevent histogram generation."""
        df = pd.DataFrame({"A": [1, np.nan, 3, 4, 5], "B": [10, 10, np.nan, 10, 10]})
        plot_ip_histograms(df)

    # --- figure properties --------------------------------------------------

    def test_figure_created(self):
        """A matplotlib Figure is created."""
        df = make_df()
        plot_ip_histograms(df)
        assert plt.get_fignums(), "No figure was created."

    def test_num_cols_one(self):
        """num_cols=1 produces a single-column grid without error."""
        df = make_df()
        plot_ip_histograms(df, num_cols=1)

    def test_figwidth_accepted(self):
        """Custom figwidth is accepted without error."""
        df = make_df()
        plot_ip_histograms(df, figwidth=20)

    # --- non-numeric columns are ignored ------------------------------------

    def test_non_numeric_columns_ignored(self):
        """String columns in df are not plotted and do not raise."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "label": ["x", "y", "z"]})
        plot_ip_histograms(df)

    # --- add_points=None is a no-op ----------------------------------------

    def test_add_points_none(self):
        """Passing add_points=None is equivalent to not providing it."""
        df = make_df()
        plot_ip_histograms(df, add_points=None)


# ===========================================================================
# plot_ip_boxplots
# ===========================================================================


class TestPlotIpBoxplots:
    """Tests for plot_ip_boxplots."""

    # --- basic smoke tests --------------------------------------------------

    def test_basic_no_add_points(self):
        """Function completes without error for a simple DataFrame."""
        df = make_df()
        plot_ip_boxplots(df)

    def test_single_column(self):
        """Function handles a single-column DataFrame."""
        df = pd.DataFrame({"X": [1, 2, 3, 4, 5]})
        plot_ip_boxplots(df, num_cols=1)

    def test_series_converted_to_frame(self):
        """A 1-D Series is promoted to a DataFrame internally."""
        s = pd.Series([1, 2, 3, 4, 5], name="Z")
        plot_ip_boxplots(s)

    def test_many_columns_grid(self):
        """Grid layout is constructed for more than num_cols columns."""
        df = pd.DataFrame(
            {"A": range(10), "B": range(10), "C": range(10), "D": range(10)}
        )
        plot_ip_boxplots(df, num_cols=2)

    # --- categorical grouping ----------------------------------------------

    def test_category_column(self):
        """Categorical grouping column produces one box per category."""
        df = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6],
                "group": ["X", "X", "X", "Y", "Y", "Y"],
            }
        )
        plot_ip_boxplots(df, category_column_name="group")

    def test_category_column_not_in_df_fallback(self):
        """A missing category column falls back to a single box per variable."""
        df = make_df()
        plot_ip_boxplots(df, category_column_name="nonexistent")

    def test_both_names_true_with_category(self):
        """With both_names=True and a valid category column, title contains 'by'."""
        df = pd.DataFrame({"A": [1, 2, 3, 4], "group": ["X", "X", "Y", "Y"]})
        with unittest.mock.patch("matplotlib.pyplot.show"):
            plot_ip_boxplots(df, category_column_name="group", both_names=True)
        fig = plt.gcf()
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert any("by" in t for t in titles)

    def test_both_names_false_with_category(self):
        """With both_names=False the title does not contain 'by'."""
        df = pd.DataFrame({"A": [1, 2, 3, 4], "group": ["X", "X", "Y", "Y"]})
        with unittest.mock.patch("matplotlib.pyplot.show"):
            plot_ip_boxplots(df, category_column_name="group", both_names=False)
        fig = plt.gcf()
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        assert all("by" not in t for t in titles)

    # --- add_points with matching names ------------------------------------

    def test_add_points_same_names(self):
        """add_points with matching column names are overlaid without error."""
        df = make_df()
        ap = make_add_points_same_names()
        plot_ip_boxplots(df, add_points=ap, add_points_col=["red", "blue"])

    def test_add_points_single_point(self):
        """A single-row add_points is accepted."""
        df = make_df()
        ap = make_add_points_one_row()
        plot_ip_boxplots(df, add_points=ap, add_points_col=["green"])

    # --- add_points with different names -----------------------------------

    def test_add_points_different_names_aligned(self):
        """add_points with different column names are silently aligned."""
        df = make_df()
        ap = make_add_points_diff_names()
        plot_ip_boxplots(df, add_points=ap, add_points_col=["red", "blue"])

    def test_add_points_three_col_different_names(self):
        """Three-column alignment works when both DataFrames have 3 numeric cols."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]})
        ap = pd.DataFrame({"p": [1.5], "q": [4.5], "r": [7.5]})
        plot_ip_boxplots(df, add_points=ap, add_points_col=["cyan"])

    # --- error cases --------------------------------------------------------

    def test_add_points_col_length_mismatch_raises(self):
        """ValueError is raised when add_points rows != add_points_col length."""
        df = make_df()
        ap = make_add_points_same_names()  # 2 rows
        with pytest.raises(ValueError, match="Length of add_points"):
            plot_ip_boxplots(df, add_points=ap, add_points_col=["red"])  # 1 colour

    # --- NaN handling -------------------------------------------------------

    def test_add_points_with_nan_skipped(self):
        """NaN entries in add_points are dropped before scatter-plotting."""
        df = make_df()
        ap = pd.DataFrame({"A": [1.5, np.nan], "B": [10.0, np.nan]})
        plot_ip_boxplots(df, add_points=ap, add_points_col=["red", "blue"])

    def test_df_with_nan_values(self):
        """NaN values in df do not prevent boxplot generation."""
        df = pd.DataFrame({"A": [1, np.nan, 3, 4, 5], "B": [10, 10, np.nan, 10, 10]})
        plot_ip_boxplots(df)

    # --- figure properties --------------------------------------------------

    def test_figure_created(self):
        """A matplotlib Figure is created."""
        df = make_df()
        plot_ip_boxplots(df)
        assert plt.get_fignums(), "No figure was created."

    def test_num_cols_one(self):
        """num_cols=1 produces a single-column grid without error."""
        df = make_df()
        plot_ip_boxplots(df, num_cols=1)

    def test_box_width_accepted(self):
        """Custom box_width is accepted without error."""
        df = make_df()
        plot_ip_boxplots(df, box_width=0.5)

    def test_height_per_subplot_accepted(self):
        """Custom height_per_subplot is accepted without error."""
        df = make_df()
        plot_ip_boxplots(df, height_per_subplot=4.0)

    # --- non-numeric columns are ignored ------------------------------------

    def test_non_numeric_columns_ignored(self):
        """String columns in df are not plotted and do not raise."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6], "label": ["x", "y", "z"]})
        plot_ip_boxplots(df)

    # --- add_points=None is a no-op ----------------------------------------

    def test_add_points_none(self):
        """Passing add_points=None is equivalent to not providing it."""
        df = make_df()
        plot_ip_boxplots(df, add_points=None)

    # --- integration: histogram + boxplot use the same alignment logic ------

    def test_histogram_and_boxplot_same_alignment(self):
        """Both functions produce the same column alignment for mismatched names."""
        df = make_df()
        ap_diff = make_add_points_diff_names()

        # Neither call should raise
        plot_ip_histograms(df, add_points=ap_diff, add_points_col=["red", "blue"])
        plot_ip_boxplots(df, add_points=ap_diff, add_points_col=["red", "blue"])
