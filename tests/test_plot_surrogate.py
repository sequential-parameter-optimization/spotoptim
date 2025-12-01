"""Tests for the plot_surrogate method of SpotOptim."""

import pytest
import numpy as np
from spotoptim import SpotOptim
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt


def sphere_function(X):
    """Simple 2D sphere function for testing."""
    return np.sum(X**2, axis=1)


def test_plot_surrogate_basic():
    """Test basic plotting functionality."""
    # Create and run optimizer
    bounds = [(-5, 5), (-5, 5)]
    opt = SpotOptim(
        fun=sphere_function, bounds=bounds, max_iter=5, n_initial=5, seed=42
    )
    result = opt.optimize()

    # Test plotting without showing
    opt.plot_surrogate(i=0, j=1, show=False)
    plt.close("all")


def test_plot_surrogate_with_names():
    """Test plotting with custom variable names."""
    bounds = [(-5, 5), (-5, 5)]
    opt = SpotOptim(
        fun=sphere_function, bounds=bounds, max_iter=5, n_initial=5, seed=42
    )
    result = opt.optimize()

    # Test with variable names
    opt.plot_surrogate(i=0, j=1, var_name=["x1", "x2"], show=False)
    plt.close("all")


def test_plot_surrogate_3d():
    """Test plotting with 3D data."""

    def sphere_3d(X):
        return np.sum(X**2, axis=1)

    bounds = [(-5, 5), (-5, 5), (-3, 3)]
    opt = SpotOptim(fun=sphere_3d, bounds=bounds, max_iter=5, n_initial=5, seed=42)
    result = opt.optimize()

    # Plot dimensions 0 and 2
    opt.plot_surrogate(i=0, j=2, show=False)
    plt.close("all")


def test_plot_surrogate_custom_params():
    """Test plotting with custom parameters."""
    bounds = [(-5, 5), (-5, 5)]
    opt = SpotOptim(
        fun=sphere_function, bounds=bounds, max_iter=5, n_initial=5, seed=42
    )
    result = opt.optimize()

    # Test with custom parameters
    opt.plot_surrogate(
        i=0,
        j=1,
        alpha=0.5,
        cmap="viridis",
        num=50,
        add_points=False,
        grid_visible=False,
        contour_levels=20,
        figsize=(10, 8),
        show=False,
    )
    plt.close("all")


def test_plot_surrogate_before_optimization():
    """Test that plotting fails before optimization is run."""
    bounds = [(-5, 5), (-5, 5)]
    opt = SpotOptim(
        fun=sphere_function, bounds=bounds, max_iter=5, n_initial=5, seed=42
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="No optimization data available"):
        opt.plot_surrogate(i=0, j=1, show=False)


def test_plot_surrogate_invalid_dimensions():
    """Test that plotting fails with invalid dimension indices."""
    bounds = [(-5, 5), (-5, 5)]
    opt = SpotOptim(
        fun=sphere_function, bounds=bounds, max_iter=5, n_initial=5, seed=42
    )
    result = opt.optimize()

    # Test with invalid i
    with pytest.raises(ValueError, match="must be less than"):
        opt.plot_surrogate(i=5, j=1, show=False)

    # Test with invalid j
    with pytest.raises(ValueError, match="must be less than"):
        opt.plot_surrogate(i=0, j=5, show=False)

    # Test with i == j
    with pytest.raises(ValueError, match="must be different"):
        opt.plot_surrogate(i=0, j=0, show=False)


def test_plot_surrogate_with_kriging():
    """Test plotting with Kriging surrogate."""
    from spotoptim import Kriging

    bounds = [(-5, 5), (-5, 5)]
    opt = SpotOptim(
        fun=sphere_function,
        bounds=bounds,
        max_iter=10,  # Need at least one sequential iteration to fit surrogate
        n_initial=5,
        surrogate=Kriging(seed=42),
        seed=42,
    )
    result = opt.optimize()

    # Test plotting with Kriging
    opt.plot_surrogate(i=0, j=1, show=False)
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
