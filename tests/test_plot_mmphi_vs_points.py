# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for plot_mmphi_vs_points in spotoptim.sampling.mm
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from spotoptim.sampling.mm import plot_mmphi_vs_points


@pytest.fixture
def base_design_2d():
    """Create a simple 2D base design."""
    return np.array([[0.1, 0.2], [0.4, 0.5], [0.7, 0.8]])


@pytest.fixture
def bounds_2d():
    """Create 2D bounds."""
    x_min = np.array([0.0, 0.0])
    x_max = np.array([1.0, 1.0])
    return x_min, x_max


class TestPlotMmphiVsPointsBasic:
    @patch("matplotlib.pyplot.show")
    def test_basic_execution(self, mock_show, base_design_2d, bounds_2d):
        """Test basic execution with minimal parameters."""
        x_min, x_max = bounds_2d
        df_summary = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=5,
            p_max=10,
            p_step=5,
            n_repeats=2,
        )
        assert isinstance(df_summary, pd.DataFrame)
        assert "n_points" in df_summary.columns
        assert "mmphi" in df_summary.columns
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_returns_dataframe_structure(self, mock_show, base_design_2d, bounds_2d):
        """Test that returned DataFrame has correct structure."""
        x_min, x_max = bounds_2d
        df = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=10,
            p_max=20,
            p_step=10,
            n_repeats=3,
        )
        # Should have mean and std columns in MultiIndex
        assert df.shape[0] == 2  # Two point counts: 10 and 20
        assert "n_points" in df.columns
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_single_point_count(self, mock_show, base_design_2d, bounds_2d):
        """Test with a single point count."""
        x_min, x_max = bounds_2d
        df = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=10,
            p_max=10,
            p_step=10,
            n_repeats=3,
        )
        assert df.shape[0] == 1
        mock_show.assert_called_once()


class TestPlotMmphiVsPointsParameterVariations:
    @patch("matplotlib.pyplot.show")
    def test_different_n_repeats(self, mock_show, base_design_2d, bounds_2d):
        """Test with different number of repeats."""
        x_min, x_max = bounds_2d
        df = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=5,
            p_max=5,
            p_step=5,
            n_repeats=10,
        )
        assert isinstance(df, pd.DataFrame)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_large_step_size(self, mock_show, base_design_2d, bounds_2d):
        """Test with large step size."""
        x_min, x_max = bounds_2d
        df = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=10,
            p_max=50,
            p_step=20,
            n_repeats=2,
        )
        assert df.shape[0] == 3  # 10, 30, 50
        mock_show.assert_called_once()


class TestPlotMmphiVsPointsEdgeCases:
    @patch("matplotlib.pyplot.show")
    def test_higher_dimensional_design(self, mock_show):
        """Test with higher dimensional design."""
        X_base = np.random.rand(5, 4)  # 5 points in 4D
        x_min = np.zeros(4)
        x_max = np.ones(4)
        df = plot_mmphi_vs_points(
            X_base=X_base,
            x_min=x_min,
            x_max=x_max,
            p_min=5,
            p_max=10,
            p_step=5,
            n_repeats=2,
        )
        assert isinstance(df, pd.DataFrame)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_with_different_bounds(self, mock_show, base_design_2d):
        """Test with non-unit bounds."""
        x_min = np.array([0.2, 0.3])
        x_max = np.array([0.8, 0.9])
        df = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=5,
            p_max=5,
            p_step=5,
            n_repeats=2,
        )
        assert isinstance(df, pd.DataFrame)
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_small_base_design(self, mock_show, bounds_2d):
        """Test with minimal base design (2 points)."""
        X_base = np.array([[0.1, 0.2], [0.8, 0.9]])
        x_min, x_max = bounds_2d
        df = plot_mmphi_vs_points(
            X_base=X_base,
            x_min=x_min,
            x_max=x_max,
            p_min=5,
            p_max=5,
            p_step=5,
            n_repeats=2,
        )
        assert isinstance(df, pd.DataFrame)
        mock_show.assert_called_once()


class TestPlotMmphiVsPointsOutputValidation:
    @patch("matplotlib.pyplot.show")
    def test_mmphi_values_positive(self, mock_show, base_design_2d, bounds_2d):
        """Test that mmphi values are positive."""
        x_min, x_max = bounds_2d
        df = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=10,
            p_max=20,
            p_step=10,
            n_repeats=3,
        )
        # Check mean values are positive
        assert (df["mmphi"]["mean"] > 0).all()
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_std_values_non_negative(self, mock_show, base_design_2d, bounds_2d):
        """Test that std values are non-negative."""
        x_min, x_max = bounds_2d
        df = plot_mmphi_vs_points(
            X_base=base_design_2d,
            x_min=x_min,
            x_max=x_max,
            p_min=10,
            p_max=20,
            p_step=10,
            n_repeats=3,
        )
        # Standard deviation should be non-negative
        assert (df["mmphi"]["std"] >= 0).all()
        mock_show.assert_called_once()
