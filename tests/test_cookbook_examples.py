# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim
from spotoptim.function.so import robot_arm_hard

def obj_fun(X):
    return np.sum(X**2, axis=1)

def test_example_1_default_de():
    """Test Example 1: Default Configuration (Differential Evolution)."""
    spot = SpotOptim(
        fun=obj_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=2, # Explicitly set to satisfy max_iter >= n_initial
        acquisition="EI",
    )
    spot.optimize()
    assert spot.best_y_ is not None
    assert spot.counter >= 10

def test_example_2_custom_de():
    """Test Example 2: Customizing Differential Evolution."""
    de_kwargs = {
        "maxiter": 200,    
        "popsize": 30,     
        "mutation": (0.6, 1.1)
    }

    spot = SpotOptim(
        fun=obj_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=2,
        acquisition="EI",
        acquisition_optimizer="differential_evolution",
        acquisition_optimizer_kwargs=de_kwargs
    )
    spot.optimize()
    assert spot.best_y_ is not None

def test_example_3_lbfgs():
    """Test Example 3: Using Gradient-Based Optimization (L-BFGS-B)."""
    lbfgs_kwargs = {
        "method": "L-BFGS-B",
        "options": {
            "maxiter": 100,
            "ftol": 1e-9
        }
    }

    spot = SpotOptim(
        fun=obj_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=2,
        acquisition="EI",
        acquisition_optimizer="L-BFGS-B",
        acquisition_optimizer_kwargs=lbfgs_kwargs
    )
    spot.optimize()
    assert spot.best_y_ is not None

def test_example_4_nelder_mead():
    """Test Example 4: Using Gradient-Free Optimization (Nelder-Mead)."""
    nm_kwargs = {
        "method": "Nelder-Mead",
        "options": {
            "maxiter": 500,
            "adaptive": True
        }
    }

    spot = SpotOptim(
        fun=obj_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=2,
        acquisition="EI",
        acquisition_optimizer="Nelder-Mead",
        acquisition_optimizer_kwargs=nm_kwargs
    )
    spot.optimize()
    assert spot.best_y_ is not None

def test_example_5_multi_cands():
    """Test Example 5: Returning Multiple Candidates."""
    spot = SpotOptim(
        fun=obj_fun,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=5, 
        n_initial=2,
        acquisition="EI",
        acquisition_fun_return_size=5 
    )
    spot.optimize()
    assert spot.best_y_ is not None

def test_robot_arm_scenario():
    """Test based on 009_robot_arm.py: High-dim, parallel, defaults."""
    n_dim = 10
    bounds = [(0.0, 1.0)] * n_dim
    var_type = ["float"] * n_dim
    var_name = [f"q{i}" for i in range(n_dim)]
    
    # Match 009_robot_arm.py: Use ARD Kernel
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, ConstantKernel
    
    kernel = ConstantKernel(1.0, (1e-2, 1e12)) * Matern(
        length_scale=np.ones(n_dim), 
        length_scale_bounds=(1e-4, 1e12), 
        nu=2.5
    )
    surrogate = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10) # Reduced restarts for test speed

    # Use defaults (acquisition_optimizer_kwargs=None)
    # verify that the internal default {'maxiter': 10000, 'gtol': 1e-9} works
    opt = SpotOptim(
        fun=robot_arm_hard,
        bounds=bounds,
        var_type=var_type,
        var_name=var_name,
        surrogate=surrogate,
        max_iter=25,    # Match 009_robot_arm.py (must be >= n_initial)
        max_time=np.inf, 
        n_initial=20,    # Match 009_robot_arm.py
        seed=42,
        verbose=True,
        tensorboard_log=False,  
        acquisition_optimizer='de_tricands',
        repeats_initial=1,
        repeats_surrogate=1,
        n_jobs=2,       # Parallel, but slightly fewer workers
    )

    opt.optimize()
    
    assert opt.best_y_ is not None
    assert opt.counter >= 15
    # Check default kwargs were applied to config if None provided
    # Note: acquisition_optimizer_kwargs might be None in init arg, but handled?
    # Actually logic sets it if None.
    assert opt.config.acquisition_optimizer_kwargs["maxiter"] == 10000
    assert opt.config.acquisition_optimizer_kwargs["gtol"] == 1e-9
