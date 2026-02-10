# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from spotoptim.plot.mo import plot_mo


def test_plot_mo_basic_runs():
    target_names = ["Target 1", "Target 2"]
    combinations = [(0, 1)]
    y_rf = np.array([[1, 2], [2, 1], [3, 3]])
    plot_mo(target_names=target_names, combinations=combinations, pareto="min", y_rf=y_rf)
    plt.close("all")


def test_plot_mo_with_orig_and_best():
    target_names = ["Target 1", "Target 2"]
    combinations = [(0, 1)]
    y_rf = np.array([[1, 2], [2, 1], [3, 3]])
    y_orig = np.array([[0, 4], [4, 0], [2, 2]])
    y_best = np.array([1, 2])
    plot_mo(target_names=target_names, combinations=combinations, pareto="min", y_rf=y_rf, y_orig=y_orig, y_best=y_best)
    plt.close("all")


def test_plot_mo_with_labels_and_fronts():
    combinations = [(0, 1)]
    y_rf = np.array([[1, 2], [2, 1], [3, 3]])
    y_orig = np.array([[0, 4], [4, 0], [2, 2]])
    target_names = ["A", "B"]
    plot_mo(
        target_names=target_names,
        combinations=combinations,
        pareto="min",
        y_rf=y_rf,
        y_orig=y_orig,
        pareto_label=True,
        pareto_front=True,
        pareto_front_orig=True,
        title="Test Plot",
    )
    plt.close("all")


def test_plot_mo_empty_inputs():
    combinations = [(0, 1)]
    target_names = ["Target 1", "Target 2"]
    plot_mo(target_names=target_names, combinations=combinations, pareto="min")
    plt.close("all")
