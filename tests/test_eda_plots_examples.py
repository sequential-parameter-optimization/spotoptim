# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
import matplotlib.pyplot as plt
import pytest
from spotoptim.eda.plots import plot_ip_histograms, plot_ip_boxplots


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib plots after each test to avoid memory issues and GUI popups."""
    yield
    plt.close("all")


def test_plot_ip_histograms_example():
    """Test the example in plot_ip_histograms docstring."""
    data = {"A": [1, 2, 2, 3, 4, 5, 100], "B": [10, 10, 10, 10, 10, 10, 10]}
    df = pd.DataFrame(data)

    # Mock plt.show to avoid blocking
    import unittest.mock

    with unittest.mock.patch("matplotlib.pyplot.show"):
        # First call from example
        plot_ip_histograms(df, bins=5, num_cols=1, thrs_unique=3)

        # Second call with add_points
        add_points = pd.DataFrame({"A": [1.5, 3.5], "B": [10, 10]})
        plot_ip_histograms(df, add_points=add_points, add_points_col=["red", "blue"])


def test_plot_ip_boxplots_example():
    """Test the example in plot_ip_boxplots docstring."""
    data = {"A": [1, 2, 2, 3, 4, 5, 100], "B": [10, 10, 10, 10, 10, 10, 10]}
    df = pd.DataFrame(data)

    import unittest.mock

    with unittest.mock.patch("matplotlib.pyplot.show"):
        # First call from example
        plot_ip_boxplots(df, num_cols=1)

        # Second call with add_points
        add_points = pd.DataFrame({"A": [1.5, 3.5], "B": [10, 10]})
        plot_ip_boxplots(df, add_points=add_points, add_points_col=["red", "blue"])
