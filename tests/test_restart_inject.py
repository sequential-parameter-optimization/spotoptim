# SPDX-FileCopyrightText: 2026 bartzbeielstein
#
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pytest
from spotoptim import SpotOptim

def test_restart_inject_best():
    """Verify that restart_inject_best=True uses the best point from the previous run."""
    
    # Define a simple objective function
    # We want it to be easy to find a "best" point, but then stagnate to trigger restart
    # Let's say the global minimum is at [0, 0]
    
    class MockObj:
        def __init__(self):
            self.counter = 0
            self.called_with_best = False
            self.best_x_found = None
            
        def __call__(self, X):
            y = []
            for x in X:
                self.counter += 1
                
                # Check if we are being called with the best point (approx [0,0])
                if self.best_x_found is not None:
                    dist = np.linalg.norm(x - self.best_x_found)
                    if dist < 1e-6:
                        self.called_with_best = True
                
                # Simple bowl
                val = np.sum(x**2)
                y.append(val)
                
                # After some iterations, return garbage to force stagnation/restart
                # But allowing enough initial progress to find a "best"
                if self.counter > 30 and self.counter < 100:
                     # Stagnation phase to trigger restart
                     y[-1] = 1000.0 + self.counter # Bad value, no improvement
            
            return np.array(y)

    obj = MockObj()
    
    # Setup optimizer
    # success_rate window needs to be small to trigger restart
    # restart_after_n=5 to trigger restart quickly
    # n_initial=5
    # max_iter=20 (enough for restart)
    optimizer = SpotOptim(
        fun=obj,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=40,
        n_initial=5,
        restart_after_n=2,
        restart_inject_best=True,
        seed=42,
        verbose=True,
        max_surrogate_points=10,
    )
    
    # Manually seed a "best" point in the first run to ensure we have something known
    # Actually, let's just let it run. The first run should find something reasonably good (better than 1000).
    # Then it will stagnate.
    # The restart should initialize with that best point.
    
    optimizer.optimize()
    
    # We can check if `optimizer.restarts_results_` has > 1 entries
    assert len(optimizer.restarts_results_) > 1, "Optimization should have restarted"
    
    # Get best from first run
    best_res_Run1 = optimizer.restarts_results_[0]
    obj.best_x_found = best_res_Run1.x
    print(f"Run 1 Best: {best_res_Run1.x}, Fun: {best_res_Run1.fun}")
    
    # Check if the second run started with this point
    # We can inspect the X_ of the second run if we could access it easily.
    # But `optimize` accumulates results. 
    # However, our MockObj checks if it was called with `best_x_found`.
    # Since `optimize` loops, the second run calls `_optimize_single_run(..., X0=best_x)`
    # which calls `get_initial_design(X0)`, which puts it in the initial design.
    # So `obj` should see it.
    
    # Note: `called_with_best` might be True from Run 1 itself if it re-evaluated it? 
    # No, usually we don't re-evaluate unless noise handling.
    # But to be sure, we can reset the flag after Run 1 if we could pause.
    
    # Alternatively, we can check logs or spy on `_optimize_single_run`.
    # But let's trust the logic: if `restart_inject_best` is working, X0 passed to run 2 is best_res.x.
    
    # Let's verify simply that the first point evaluated in Run 2 (or in initial design of Run 2) matches best of Run 1.
    # We can't easily see "Run 2" explicitly in `optimizer` object after the fact except in `restarts_results_`.
    # But `restarts_results_` contains `X` and `y` for each run.
    
    run2_res = optimizer.restarts_results_[1]
    run2_X = run2_res.X
    
    # Check if the first point in Run 2 is the best point from Run 1
    # Allow for some small numerical diffs
    first_point_Run2 = run2_X[0]
    best_point_Run1 = best_res_Run1.x
    
    dist = np.linalg.norm(first_point_Run2 - best_point_Run1)
    print(f"Run 2 First Point: {first_point_Run2}")
    print(f"Run 1 Best Point:  {best_point_Run1}")
    print(f"Distance: {dist}")
    
    assert dist < 1e-6, "Run 2 should start with the best point from Run 1"

if __name__ == "__main__":
    test_restart_inject_best()
