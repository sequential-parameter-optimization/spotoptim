# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
from spotoptim import SpotOptim
from sklearn.gaussian_process import GaussianProcessRegressor
from functools import partial

def objective_fun(X):
    return np.sum(X**2, axis=1)

def test_acquisition_kwargs_config():
    """Verify acquisition_optimizer_kwargs is correctly stored in config."""
    control = {"method": "L-BFGS-B", "options": {"maxiter": 100, "gtol": 1e-4}}
    optimizer = SpotOptim(
        fun=objective_fun,
        bounds=[(-5, 5)],
        max_iter=5,
        n_initial=2,
        acquisition_optimizer_kwargs=control
    )
    
    assert optimizer.config.acquisition_optimizer_kwargs == control
    assert isinstance(optimizer.surrogate, GaussianProcessRegressor)
    
    # Verify optimizer is a partial function
    assert isinstance(optimizer.surrogate.optimizer, partial)
    # Check that kwargs match what we passed
    assert optimizer.surrogate.optimizer.keywords == control

def test_acquisition_kwargs_default():
    """Verify default behavior (no kwargs)."""
    optimizer = SpotOptim(
        fun=objective_fun,
        bounds=[(-5, 5)],
        max_iter=5,
        n_initial=2
    )
    
    assert optimizer.config.acquisition_optimizer_kwargs == {'maxiter': 10000, 'gtol': 1e-9}
    # Should use partial wrapper with default kwargs
    assert isinstance(optimizer.surrogate.optimizer, partial)
    assert optimizer.surrogate.optimizer.keywords == {'maxiter': 10000, 'gtol': 1e-9}


def test_custom_options_execution(capsys):
    """Run optimization with L-BFGS-B and custom options."""
    control = {"options": {"maxiter": 5}} 
    optimizer = SpotOptim(
        fun=objective_fun,
        bounds=[(-1, 1)],
        max_iter=5, 
        n_initial=3,
        acquisition_optimizer_kwargs=control,
        seed=42
    )
    
    optimizer.optimize()
    
    assert optimizer.counter > 0
    assert optimizer.best_y_ is not None

def test_nelder_mead_execution(capsys):
    """Run optimization with Nelder-Mead (gradient-free)."""
    # this tests the gradient-stripping wrapper logic
    control = {"method": "Nelder-Mead", "options": {"maxiter": 100}}
    optimizer = SpotOptim(
        fun=objective_fun,
        bounds=[(-1, 1)],
        max_iter=5,
        n_initial=3,
        acquisition_optimizer_kwargs=control,
        seed=42
    )
    
    optimizer.optimize()
    
    assert optimizer.counter > 0
    assert optimizer.best_y_ is not None
