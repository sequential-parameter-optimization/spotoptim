# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim

def dummy_obj(X):
    return np.sum(X**2, axis=1)

def test_n_initial_zero_with_x0():
    print("Testing n_initial=0 with x0 provided...")
    
    # Define 3 initial points
    X_start = np.array([
        [0.1, 0.1],
        [0.5, 0.5],
        [0.9, 0.9]
    ])
    
    # Initialize SpotOptim with n_initial=0 but providing x0
    optimizer = SpotOptim(
        fun=dummy_obj,
        bounds=[(0, 1), (0, 1)],
        n_initial=0,
        max_iter=5, # Allow some iterations
        x0=X_start,
        seed=42,
        surrogate=None # Use default GP
    )
    
    # Optimization should run
    # Initial design will just be x0 (3 points)
    # Then it will proceed to sequential steps until max_iter=5 reached
    # Since we have 3 points already, it can do 2 more steps (total 5 evals)
    
    optimizer.optimize()
    
    X_evaluated = optimizer.X_
    print(f"Evaluated X shape: {X_evaluated.shape}")
    print("First 3 points:")
    print(X_evaluated[:3])
    
    # First 3 points should match X_start
    np.testing.assert_allclose(X_evaluated[:3], X_start, atol=1e-6)
    
    # Total evaluations should be max_iter (5)
    assert X_evaluated.shape[0] == 5
    
    print("PASSED: test_n_initial_zero_with_x0")

if __name__ == "__main__":
    test_n_initial_zero_with_x0()
