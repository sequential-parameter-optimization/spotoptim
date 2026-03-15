# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotoptim.plot.visualization module."""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spotoptim import SpotOptim
from spotoptim.plot.visualization import (
    plot_surrogate,
    plot_progress,
    plot_important_hyperparameter_contour,
    plot_design_points,
)

# Use non-interactive backend for testing
matplotlib.use("Agg")


@pytest.fixture
def optimizer():
    """Create a trained SpotOptim instance for testing."""

    def sphere(X):
        return np.sum(X**2, axis=1)

    opt = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5)] * 4,
        max_iter=15,
        n_initial=10,
        var_name=["x1", "x2", "x3", "x4"],
        seed=42,
    )
    opt.optimize()
    return opt


def test_plot_surrogate(optimizer):
    """Test plot_surrogate function."""
    try:
        plot_surrogate(optimizer, i=0, j=1, show=False)
        plt.close("all")
    except Exception as e:
        pytest.fail(f"plot_surrogate raised an exception: {e}")


def test_plot_progress(optimizer):
    """Test plot_progress function."""
    try:
        plot_progress(optimizer, show=False)
        plt.close("all")

        plot_progress(optimizer, show=False, log_y=True)
        plt.close("all")
    except Exception as e:
        pytest.fail(f"plot_progress raised an exception: {e}")


def test_plot_important_hyperparameter_contour(optimizer):
    """Test plot_important_hyperparameter_contour function."""
    try:
        plot_important_hyperparameter_contour(optimizer, max_imp=2, show=False)
        plt.close("all")
    except Exception as e:
        pytest.fail(f"plot_important_hyperparameter_contour raised an exception: {e}")


class TestPlotDesignPoints:
    """Tests for plot_design_points."""

    def setup_method(self):
        self.X2d = np.array([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [1.0, 2.0]])
        self.X4d = np.random.default_rng(0).uniform(0, 1, size=(20, 4))

    # --- basic 2-D case ---

    def test_returns_figure_2d(self):
        fig = plot_design_points(self.X2d, show=False)
        assert hasattr(fig, "savefig")
        plt.close("all")

    def test_2d_default_dims(self):
        fig = plot_design_points(self.X2d, show=False, figsize=(4, 4))
        assert fig is not None
        plt.close("all")

    def test_2d_custom_dims(self):
        X = np.column_stack([self.X2d, self.X2d[:, 0]])  # 3-D
        fig = plot_design_points(X, i=0, j=2, agg="mean", show=False, figsize=(4, 4))
        assert fig is not None
        plt.close("all")

    # --- higher-dimensional cases ---

    def test_4d_mean_agg(self):
        fig = plot_design_points(
            self.X4d, i=0, j=1, agg="mean", show=False, figsize=(5, 5)
        )
        assert fig is not None
        plt.close("all")

    def test_4d_median_agg(self):
        fig = plot_design_points(self.X4d, i=1, j=3, agg="median", show=False)
        assert fig is not None
        plt.close("all")

    def test_4d_min_agg(self):
        fig = plot_design_points(self.X4d, i=0, j=2, agg="min", show=False)
        assert fig is not None
        plt.close("all")

    def test_4d_max_agg(self):
        fig = plot_design_points(self.X4d, i=0, j=3, agg="max", show=False)
        assert fig is not None
        plt.close("all")

    def test_callable_agg(self):
        fig = plot_design_points(self.X4d, i=0, j=1, agg=np.std, show=False)
        assert fig is not None
        plt.close("all")

    # --- colorbar present only for n_dim > 2 ---

    def test_colorbar_added_for_hidden_dims(self):
        fig = plot_design_points(self.X4d, i=0, j=1, agg="mean", show=False)
        # A colorbar produces an Axes with label
        ax_labels = [ax.get_label() for ax in fig.get_axes()]
        assert any("colorbar" in lbl for lbl in ax_labels) or len(fig.get_axes()) >= 2
        plt.close("all")

    def test_no_colorbar_for_2d(self):
        fig = plot_design_points(self.X2d, show=False)
        # Only one Axes for the scatter (no colorbar axis)
        assert len(fig.get_axes()) == 1
        plt.close("all")

    # --- SpotOptim integration ---

    def test_with_spotoptim_get_initial_design(self):
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5), (-5, 5)],
            n_initial=10,
            seed=0,
        )
        X0 = opt.get_initial_design()
        fig = plot_design_points(X0, i=0, j=1, show=False, figsize=(5, 4))
        assert fig is not None
        plt.close("all")

    def test_with_spotoptim_4d(self):
        opt = SpotOptim(
            fun=lambda X: np.sum(X**2, axis=1),
            bounds=[(-5, 5)] * 4,
            n_initial=15,
            seed=1,
        )
        X0 = opt.get_initial_design()
        fig = plot_design_points(X0, i=0, j=1, agg="mean", show=False, figsize=(6, 5))
        assert fig is not None
        plt.close("all")

    # --- kwargs forwarded to scatter ---

    def test_scatter_kwargs_forwarded(self):
        fig = plot_design_points(self.X2d, show=False, s=100, alpha=0.5, figsize=(4, 4))
        assert fig is not None
        plt.close("all")

    # --- validation errors ---

    def test_invalid_1d_input(self):
        with pytest.raises(ValueError, match="2-D array"):
            plot_design_points(np.array([1.0, 2.0, 3.0]))

    def test_invalid_single_column(self):
        with pytest.raises(ValueError, match="at least 2 columns"):
            plot_design_points(np.array([[1.0], [2.0]]))

    def test_i_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            plot_design_points(self.X2d, i=5, j=0)

    def test_j_out_of_range(self):
        with pytest.raises(ValueError, match="out of range"):
            plot_design_points(self.X2d, i=0, j=5)

    def test_i_equals_j(self):
        with pytest.raises(ValueError, match="different dimensions"):
            plot_design_points(self.X2d, i=1, j=1)

    def test_invalid_agg_string(self):
        with pytest.raises(ValueError, match="Unsupported agg"):
            plot_design_points(self.X4d, agg="sum")

    def test_invalid_agg_type(self):
        with pytest.raises(TypeError, match="string or callable"):
            plot_design_points(self.X4d, agg=42)


def test_plot_functions_validation():
    """Test validation in plotting functions."""

    def sphere(X):
        return np.sum(X**2, axis=1)

    # Optimizer without results
    opt = SpotOptim(fun=sphere, bounds=[(-1, 1)] * 2, max_iter=5, n_initial=2)
    # Don't run optimize()

    with pytest.raises(ValueError, match="No optimization data available"):
        plot_surrogate(opt, i=0, j=1)

    with pytest.raises(ValueError, match="No optimization data available"):
        plot_progress(opt)
