
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from spotoptim.eda.plots import plot_ip_histograms, plot_ip_boxplots

@pytest.fixture
def sample_df():
    return pd.DataFrame({'A': [1, 2, 3, 4, 5], 'B': [10, 20, 30, 40, 50]})

@pytest.fixture
def add_points_2():
    return pd.DataFrame({'A': [1.5, 3.5], 'B': [15, 35]})

@pytest.fixture
def add_points_1():
    return pd.DataFrame({'A': [2.5], 'B': [25]})

class TestPlotIpHistogramsColors:
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.axes.Axes.scatter')
    def test_multi_color_valid(self, mock_scatter, mock_show, sample_df, add_points_2):
        """Test providing correct number of colors for multiple points."""
        colors = ["red", "blue"]
        plot_ip_histograms(sample_df, add_points=add_points_2, add_points_col=colors)
        
        # Verify scatter was called with c=colors (series)
        assert mock_scatter.call_count == 2 # Once for each column
        
        # Check args for first call (column A)
        _, kwargs = mock_scatter.call_args_list[0]
        # Colors passed to scatter might be the Series/array from our logic
        np.testing.assert_array_equal(kwargs['c'], colors)

    def test_length_mismatch_error(self, sample_df, add_points_2):
        """Test error when color list length mismatches point count."""
        with pytest.raises(ValueError, match="Length of add_points"):
            plot_ip_histograms(sample_df, add_points=add_points_2, add_points_col=["red"])

    @patch('matplotlib.pyplot.show')
    def test_default_one_point(self, mock_show, sample_df, add_points_1):
        """Test default behavior works for single point."""
        # Default add_points_col is ["red"], len=1. add_points_1 has len=1.
        plot_ip_histograms(sample_df, add_points=add_points_1)
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.axes.Axes.scatter')
    def test_nan_handling_colors(self, mock_scatter, mock_show, sample_df):
        """Test that colors align correctly when NaNs are present."""
        # Point 1: present in A & B. Point 2: NaN in A, present in B.
        add_points_nan = pd.DataFrame({'A': [1.5, np.nan], 'B': [15, 35]})
        colors = ["red", "blue"]
        
        plot_ip_histograms(sample_df, add_points=add_points_nan, add_points_col=colors)
        
        # Call for Col A: should only plot the first point ("red")
        # Call for Col B: should plot both ("red", "blue")
        
        # Check call for A (first call likely)
        call_args_A = mock_scatter.call_args_list[0]
        # Verify only one point plotted
        assert len(call_args_A[0][0]) == 1 
        # Verify color is red
        np.testing.assert_array_equal(call_args_A[1]['c'], ["red"])

        # Check call for B (second call likely)
        call_args_B = mock_scatter.call_args_list[1]
        assert len(call_args_B[0][0]) == 2
        np.testing.assert_array_equal(call_args_B[1]['c'], ["red", "blue"])


class TestPlotIpBoxplotsColors:

    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.axes.Axes.scatter')
    def test_multi_color_valid(self, mock_scatter, mock_show, sample_df, add_points_2):
        colors = ["green", "yellow"]
        plot_ip_boxplots(sample_df, add_points=add_points_2, add_points_col=colors)
        assert mock_scatter.call_count == 2
        _, kwargs = mock_scatter.call_args_list[0]
        np.testing.assert_array_equal(kwargs['c'], colors)

    def test_length_mismatch_error(self, sample_df, add_points_2):
        with pytest.raises(ValueError, match="Length of add_points"):
            plot_ip_boxplots(sample_df, add_points=add_points_2, add_points_col=["red"])

