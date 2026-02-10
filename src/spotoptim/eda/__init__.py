# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Exploratory Data Analysis (EDA) module for spotoptim.

This module provides visualization and analysis tools for exploring
optimization results and data distributions.
"""

from spotoptim.eda.plots import plot_ip_histograms, plot_ip_boxplots

__all__ = [
    "plot_ip_histograms",
    "plot_ip_boxplots",
]
