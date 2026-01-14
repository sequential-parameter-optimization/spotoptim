
"""Tests for spotoptim.plot.visualization module."""

import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from spotoptim import SpotOptim
from spotoptim.plot.visualization import (
    plot_surrogate,
    plot_progress,
    plot_important_hyperparameter_contour
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
        var_name=['x1', 'x2', 'x3', 'x4'],
        seed=42
    )
    opt.optimize()
    return opt

def test_plot_surrogate(optimizer):
    """Test plot_surrogate function."""
    try:
        plot_surrogate(optimizer, i=0, j=1, show=False)
        plt.close('all')
    except Exception as e:
        pytest.fail(f"plot_surrogate raised an exception: {e}")

def test_plot_progress(optimizer):
    """Test plot_progress function."""
    try:
        plot_progress(optimizer, show=False)
        plt.close('all')
        
        plot_progress(optimizer, show=False, log_y=True)
        plt.close('all')
    except Exception as e:
        pytest.fail(f"plot_progress raised an exception: {e}")

def test_plot_important_hyperparameter_contour(optimizer):
    """Test plot_important_hyperparameter_contour function."""
    try:
        plot_important_hyperparameter_contour(optimizer, max_imp=2, show=False)
        plt.close('all')
    except Exception as e:
        pytest.fail(f"plot_important_hyperparameter_contour raised an exception: {e}")

def test_plot_functions_validation():
    """Test validation in plotting functions."""
    def sphere(X):
         return np.sum(X**2, axis=1)
         
    # Optimizer without results
    opt = SpotOptim(fun=sphere, bounds=[(-1, 1)]*2, max_iter=5, n_initial=2)
    # Don't run optimize()
    
    with pytest.raises(ValueError, match="No optimization data available"):
        plot_surrogate(opt, i=0, j=1)

    with pytest.raises(ValueError, match="No optimization data available"):
        plot_progress(opt)
