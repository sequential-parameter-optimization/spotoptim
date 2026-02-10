# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim

def objective_with_args(x, shift, scale, mode="default"):
    """
    Objective function with extra arguments.
    f(x) = sum( ( (x - shift) / scale )^2 )
    
    If mode is "inverse", return -f(x).
    """
    val = np.sum(((x - shift) / scale) ** 2, axis=1)
    if mode == "inverse":
        return -val
    return val

def test_args_only():
    """Test passing positional arguments to the objective function."""
    bounds = [(-5, 5), (-5, 5)]
    shift = 1.0
    scale = 2.0
    
    # Initialize SpotOptim with args only
    optimizer = SpotOptim(
        fun=objective_with_args,
        bounds=bounds,
        max_iter=15,
        n_initial=5,
        seed=42,
        args=(shift, scale)
    )
    
    result = optimizer.optimize()
    
    # Check if result is close to optimal [1.0, 1.0]
    error = np.linalg.norm(result.x - np.array([1.0, 1.0]))
    assert error < 0.1, f"Optimization failed for args only. Error: {error}"
    assert np.isclose(result.fun, 0.0, atol=0.1)

def test_kwargs_only():
    """Test passing keyword arguments to the objective function."""
    # We can use defaults for shift/scale if we don't pass them, 
    # but the function signature requires them.
    # So let's define a wrapper or use a lambda for this test if we strictly want kwargs only.
    # OR, we just pass args via kwargs if the function supported it, but it uses positional args.
    
    # Let's define a specific kwargs function
    def obj_kwargs(x, target=None):
        if target is None:
            target = np.zeros_like(x)
        return np.sum((x - target)**2, axis=1)
        
    bounds = [(-5, 5)]
    target = np.array([2.0])
    
    optimizer = SpotOptim(
        fun=obj_kwargs,
        bounds=bounds,
        max_iter=15,
        n_initial=5,
        seed=42,
        kwargs={"target": target}
    )
    
    result = optimizer.optimize()
    assert np.allclose(result.x, target, atol=0.1)

def test_args_and_kwargs():
    """Test passing both args and kwargs."""
    bounds = [(-5, 5), (-5, 5)]
    shift = 1.0
    scale = 1.0
    mode = "inverse" # maximizing equivalent to minimizing negative
    
    # objective_with_args returns -val when mode="inverse"
    # optimizing -val means maximizing val... wait.
    # minimizing (-val) is maximizing val. 
    # val = sum( ... ^2 ) >= 0.
    # -val <= 0.
    # Min of -val is -infinity if bounds allowed, or largest possible sum.
    # Max of val is at boundaries.
    
    # Let's stick to minimizing ("default" mode) but pass it explicitly via kwargs
    
    optimizer = SpotOptim(
        fun=objective_with_args,
        bounds=bounds,
        max_iter=15,
        n_initial=5,
        seed=42,
        args=(shift, scale),
        kwargs={"mode": "default"}
    )
    
    result = optimizer.optimize()
    error = np.linalg.norm(result.x - np.array([1.0, 1.0]))
    assert error < 0.1

def test_args_stored_in_object():
    """Verify that args and kwargs are stored in the SpotOptim instance."""
    def dummy(x): return np.sum(x)
    
    opt = SpotOptim(fun=dummy, bounds=[(0,1)], args=(1, 2), kwargs={"a": "b"})
    assert opt.args == (1, 2)
    assert opt.kwargs == {"a": "b"}

def test_default_args_kwargs():
    """Verify default values for args and kwargs."""
    def dummy(x): return np.sum(x)
    opt = SpotOptim(fun=dummy, bounds=[(0,1)])
    assert opt.args == ()
    assert opt.kwargs == {}
