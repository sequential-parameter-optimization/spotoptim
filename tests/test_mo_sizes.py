# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for multi-objective optimization array sizes to avoid off-by-one errors.
"""
import pytest
import numpy as np
from spotoptim import SpotOptim

def test_mo_size_consistency_with_nans():
    """
    Test that y_mo size remains consistent with X_ and y_ when NaNs are present using a multi-objective function.
    """
    def multi_obj_with_nans(X):
        # Return two objectives: sum of squares and sum of (x-1)^2
        obj1 = np.sum(X**2, axis=1)
        obj2 = np.sum((X - 1) ** 2, axis=1)
        
        # Introduce NaNs: if x[0] > 0, make it NaN
        # This will likely filter some points out
        mask = X[:, 0] > 2.0
        obj1[mask] = np.nan
        
        return np.column_stack([obj1, obj2])

    optimizer = SpotOptim(
        fun=multi_obj_with_nans,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        n_initial=10,
        seed=42,
        verbose=False
    )

    try:
        optimizer.optimize()
    except Exception:
        pass # We don't care if it fails to converge, just want to check sizes

    assert optimizer.X_ is not None
    assert optimizer.y_ is not None 
    assert optimizer.y_mo is not None

    n_X = len(optimizer.X_)
    n_y = len(optimizer.y_)
    n_y_mo = len(optimizer.y_mo)

    assert n_X == n_y, f"X_ ({n_X}) and y_ ({n_y}) sizes mismatch"
    assert n_X == n_y_mo, f"X_ ({n_X}) and y_mo ({n_y_mo}) sizes mismatch (y_mo has extra points?)"

def test_mo_size_initial_design_nan():
    """
    Test that y_mo handles NaNs correctly in the initial design phase.
    """
    def multi_obj_initial_nan(X):
        # Make the first few points NaN
        obj1 = np.sum(X**2, axis=1)
        obj2 = np.sum(X**2, axis=1)
        
        # If mean of X is small (likely early points in some ordering or just random), make it NaN
        # Here we just force some to NaN based on value to ensure filtering happens
        mask = X[:, 0] < 0 
        obj1[mask] = np.nan
        
        return np.column_stack([obj1, obj2])

    optimizer = SpotOptim(
        fun=multi_obj_initial_nan,
        bounds=[(-5, 5)],
        max_iter=10, # Short run
        n_initial=5,
        seed=42
    )
    
    optimizer.optimize()
    
    assert len(optimizer.X_) == len(optimizer.y_) == len(optimizer.y_mo)


def test_mo_size_sequential_nan():
    """
    Test that y_mo handles NaNs correctly during sequential optimization.
    """
    def multi_obj_seq_nan(X):
        obj1 = np.sum(X**2, axis=1)
        obj2 = np.sum(X**2, axis=1)
        # No NaNs initially
        return np.column_stack([obj1, obj2])

    optimizer = SpotOptim(
        fun=multi_obj_seq_nan,
        bounds=[(-5, 5)],
        max_iter=10,
        n_initial=5,
        seed=42
    )
    
    # Initialize normally
    optimizer._init_storage(np.zeros((5, 1)), np.zeros(5))
    optimizer.y_mo = np.zeros((5, 2))
    
    # Simulate a step where a point returns NaN
    x_new = np.array([[10.0]]) # Transformed space
    y_new_val = np.array([np.nan])
    y_mo_new = np.array([[np.nan, np.nan]])

    # Manually append to y_mo as _evaluate_function would do
    # Note: in real flow, _evaluate_function calls _mo2so which calls _store_mo
    # We simulate this state: y_mo has the invalid point, but we feed Nan to remove_nan logic
    optimizer.y_mo = np.vstack([optimizer.y_mo, y_mo_new])
    
    
    # Call internal handler
    x_clean, y_clean = optimizer._handle_NA_new_points(x_new, y_new_val)
    
    # Check consistency
    if x_clean is None:
        # If skipped, y_mo should be original size (5)
        assert len(optimizer.y_mo) == 5, f"y_mo should be 5 when skipped, got {len(optimizer.y_mo)}"
    else:
        # If kept (penalized), y_mo should be size 6
        assert len(optimizer.y_mo) == 6, f"y_mo should be 6 when kept, got {len(optimizer.y_mo)}"
        assert len(x_clean) == 1
        assert len(y_clean) == 1

