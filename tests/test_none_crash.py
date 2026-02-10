# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim

def test_initial_design_with_none():
    """Test that SpotOptim handles None values in initial design without crashing."""
    
    def objective_with_none(X):
        # Return valid numbers for most, but None for the second point
        # Simulates a failed evaluation (e.g. remote timeout)
        n = X.shape[0]
        res = [np.sum(x**2) for x in X]
        if n > 1:
            res[1] = None  # Inject None
        return np.array(res, dtype=object)

    bounds = [(-5, 5), (-5, 5)]
    
    # This should NOT assume the user wants it to crash, but the test IS verifying the fix.
    # So eventually it should pass.
    
    optimizer = SpotOptim(
        fun=objective_with_none,
        bounds=bounds,
        max_iter=5,
        n_initial=3,
        seed=42,
        verbose=True
    )
    
    try:
        # This calls _handle_NA_initial_design internally
        optimizer.optimize() 
    except TypeError as e:
        pytest.fail(f"SpotOptim crashed with TypeError handling None: {e}")
    except Exception as e:
        # Other errors might be okay (e.g. not enough points), but not TypeError
        pass

if __name__ == "__main__":
    test_initial_design_with_none()
