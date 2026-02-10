# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim

def test_get_best_hyperparameters_basic():
    """Test basic functionality with continuous variables."""
    # Simple quadratic function
    def fun(X):
        return np.sum(X**2, axis=1)

    opt = SpotOptim(
        fun=fun,
        bounds=[(-5, 5), (-5, 5)],
        var_name=["x", "y"],
        max_iter=10,
        n_initial=5,
        seed=42
    )
    opt.optimize()
    
    params = opt.get_best_hyperparameters()
    assert isinstance(params, dict)
    assert "x" in params
    assert "y" in params
    # Optimum is at 0,0
    assert abs(params["x"]) < 2.0
    assert abs(params["y"]) < 2.0

def test_get_best_hyperparameters_noisy():
    """Test that noisy optimization returns the best mean, not necessarily best single observation."""
    
    # Function with noise: Deterministic part + noise
    # We cheat a bit to ensure mean distinguishes them
    # Setup: 2 points. Point A: mean 10, var 1. Point B: mean 5, var 1.
    # We want Point B to be selected.
    # However, SpotOptim runs the function. We need a mock or a controlled function.
    
    # Let's use a function that returns high value for first call, low for second?
    # No, SpotOptim randomizes order.
    
    # Better: Use a mock function where we can control returns for specific inputs?
    # Or just rely on SpotOptim logic: min_mean_X is set when repeats_initial > 1.
    
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1) + np.random.normal(0, 0.1, size=len(X)), # noisy sphere
        bounds=[(-1, 1)],
        repeats_initial=2,
        max_iter=4, # Just initial design basically
        n_initial=2,
        seed=42
    )
    opt.optimize()
    
    assert opt.repeats_initial > 1
    assert hasattr(opt, "min_mean_X")
    
    params = opt.get_best_hyperparameters()
    # It should correspond to min_mean_X
    best_mean_val = opt.min_mean_X[0]
    assert params["x0"] == best_mean_val


def test_get_best_hyperparameters_types():
    """Test handling of int and factor types."""
    
    categories = ["cat", "dog", "fish"]
    
    def fun(X):
        # validation of inputs is enough here, dummy metric
        return np.sum(X[:, :2], axis=1) # only use first 2 cols

    opt = SpotOptim(
        fun=fun,
        bounds=[(-5, 5), (0, 10), tuple(categories)], # float, int, factor (strings)
        var_type=["float", "int", "factor"],
        var_name=["f", "i", "c"],
        max_iter=5,
        n_initial=5
    )
    
    # Manually populate results to test mapping without running full optim loop which might not respect bounds strictly in random initial without proper constraints logic if not using ParameterSet wrapper (but SpotOptim handles basic limits).
    # We will just run optimize() to init everything, then mock X_ and best_x_
    opt.optimize()
    
    # Mock best solution
    # f=1.5, i=5.2 (should round to 5), c=1.1 (should round to 1 -> "dog")
    opt.best_x_ = np.array([1.5, 5.2, 1.1])
    # Force noise=False to ensure best_x_ is used - not needed as repeats=1 by default
    
    params = opt.get_best_hyperparameters()
    
    assert params["f"] == 1.5
    assert isinstance(params["i"], int)
    assert params["i"] == 5
    assert params["c"] == "dog"

def test_get_best_hyperparameters_no_names():
    """Test dealing with missing var_names."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X, axis=1),
        bounds=[(0, 1), (0, 1)],
        max_iter=2,
        n_initial=2
    )
    opt.optimize()
    
    params = opt.get_best_hyperparameters()
    assert "x0" in params
    assert "x1" in params

def test_get_best_hyperparameters_as_array():
    """Test returning raw array."""
    opt = SpotOptim(
        fun=lambda X: np.sum(X, axis=1),
        bounds=[(0, 1)],
        max_iter=2,
        n_initial=2
    )
    opt.optimize()
    
    params = opt.get_best_hyperparameters(as_dict=False)
    assert isinstance(params, np.ndarray)

