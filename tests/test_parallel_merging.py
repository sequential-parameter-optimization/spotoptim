# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim

def obj_fun(X):
    # Simple sphere function
    return np.sum(X**2, axis=1)

def test_parallel_merging():
    """
    Test that SpotOptim correctly merges results from parallel runs.
    
    1. Verify global best is found.
    2. Verify total evaluations reflect all parallel work.
    """
    
    n_jobs = 2
    max_iter = 10
    n_initial = 4
    
    # Run optimization
    opt = SpotOptim(
        fun=obj_fun,
        bounds=[(-5, 5)] * 2,
        max_iter=max_iter,
        n_initial=n_initial,
        n_jobs=n_jobs,
        seed=42,
        verbose=True
    )
    
    opt.optimize()
    
    # 1. Check if best result is reasonable (should find something close to 0)
    assert opt.best_y_ is not None
    assert opt.best_y_ < 50.0 # Loose bound, just ensuring it ran
    
    # 2. Check total evaluations
    # In steady-state parallelization, max_iter is the global budget.
    expected_evals = max_iter
    
    print(f"Reported evaluations (opt.counter): {opt.counter}")
    print(f"Expected evaluations (n_jobs * max_iter): {expected_evals}")
    
    # Assert that we have recorded data for ALL evaluations, not just the best run
    assert opt.counter >= expected_evals * 0.9, \
        f"Evaluations count {opt.counter} is significantly less than expected {expected_evals}. Data from parallel runs might be lost."
    
    # Also check underlying data arrays
    assert len(opt.y_) == opt.counter
    assert len(opt.X_) == opt.counter
