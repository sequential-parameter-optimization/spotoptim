# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import numpy as np
import os
import dill
from spotoptim import SpotOptim

def global_objective(X):
    return np.sum(X**2, axis=1)

def test_save_load_lambda_function(tmp_path):
    """Test that a SpotOptim instance with a lambda function can be saved and loaded."""
    # Define optimizer with lambda
    opt = SpotOptim(
        fun=lambda X: np.sum(X**2, axis=1),
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=5,
        seed=42
    )
    
    # Run optimization
    opt.optimize()
    
    # Save result
    save_path = tmp_path / "lambda_res.pkl"
    opt.save_result(filename=str(save_path))
    
    # Load result
    loaded_opt = SpotOptim.load_result(str(save_path))
    
    # Verify function is preserved and callable
    assert loaded_opt.fun is not None
    
    # Test function execution
    X_test = np.array([[1.0, 2.0]])
    y_test = loaded_opt.fun(X_test)
    assert np.isclose(y_test[0], 5.0)

def test_save_load_local_function(tmp_path):
    """Test that a SpotOptim instance with a locally defined function can be saved and loaded."""
    
    def local_objective(X):
        return np.sum(np.abs(X), axis=1)
        
    # Define optimizer with local function
    opt = SpotOptim(
        fun=local_objective,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=5,
        seed=42
    )
    
    # Save experiment (without running)
    save_path = tmp_path / "local_exp.pkl"
    opt.save_experiment(filename=str(save_path))
    
    # Load experiment
    loaded_opt = SpotOptim.load_experiment(str(save_path))
    
    # Verify function is preserved
    assert loaded_opt.fun is not None
    
    # Test function execution
    X_test = np.array([[-1.0, -2.0]])
    y_test = loaded_opt.fun(X_test)
    assert np.isclose(y_test[0], 3.0)

def test_save_load_global_function(tmp_path):
    """Test standard global function (sanity check)."""
    opt = SpotOptim(
        fun=global_objective,
        bounds=[(-5, 5)],
        max_iter=5,
        n_initial=2
    )
    
    save_path = tmp_path / "global_res.pkl"
    opt.save_result(filename=str(save_path))
    
    loaded_opt = SpotOptim.load_result(str(save_path))
    assert loaded_opt.fun is not None
    assert loaded_opt.fun(np.array([[2.0]])) == 4.0
