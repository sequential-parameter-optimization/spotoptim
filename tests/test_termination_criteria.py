
import pytest
import numpy as np
import math
import time
from spotoptim.SpotOptim import SpotOptim

def sphere_1d(x):
    """Simple 1D sphere function."""
    return np.sum(x**2, axis=1)

def count_evals_wrapper(fun):
    """Wrapper to count function evaluations."""
    count = 0
    def wrapper(x):
        nonlocal count
        count += len(x)
        return fun(x)
    return wrapper, lambda: count

class TestTerminationCriteria:

    def test_finite_iter_infinite_time(self):
        """Test Case 1: Finite max_iter, Infinite max_time. Should stop at max_iter."""
        max_iter = 30
        opt = SpotOptim(
            fun=sphere_1d,
            bounds=[(-5, 5)],
            max_iter=max_iter,
            max_time=np.inf,
            n_initial=10,
            verbose=False
        )
        res = opt.optimize()
        assert res.nfev == max_iter, f"Expected {max_iter} evaluations, got {res.nfev}"
        assert res.success is True or res.message.startswith("Optimization terminated: maximum evaluations")

    def test_infinite_iter_finite_time(self):
        """Test Case 2: Infinite max_iter, Short max_time. Should stop by time."""
        # Use a very short time limit (e.g., 2 seconds)
        # Assuming the system is fast enough to do > 20 evals in 2 seconds.
        max_time_min = 2.0 / 60.0 # 2 seconds
        
        start_time = time.time()
        opt = SpotOptim(
            fun=sphere_1d,
            bounds=[(-5, 5)],
            max_iter=np.inf,
            max_time=max_time_min,
            n_initial=10,
            verbose=False
        )
        res = opt.optimize()
        elapsed = time.time() - start_time
        
        # Check if it ran for at least roughly the requested time (allow small margin)
        assert elapsed >= 1.5, f"Optimization stopped too early: {elapsed}s vs expected ~2s"
        
        # Should have done more than initial design + default iter (20) if valid
        # sphere_1d is non-expensive, should be very fast.
        assert res.nfev > 10, f"Expected >10 evaluations (at least initial design + 1), got {res.nfev}"
        assert "time limit" in res.message or "Time limit" in res.message, f"Expected time limit message, got: {res.message}"
        
        # Check message or status if possible (SpotOptim might return success=False if max_iter not reached but time is?)
        # Current logic: stops loop if time exceeded.
        
    def test_math_inf_compatibility(self):
        """Test Case 4: explicit math.inf support check (vs np.inf)."""
        max_iter = 30
        opt = SpotOptim(
            fun=sphere_1d,
            bounds=[(-5, 5)],
            max_iter=max_iter,
            max_time=math.inf, # Using math.inf instead of np.inf
            n_initial=10,
            verbose=False
        )
        res = opt.optimize()
        assert res.nfev == max_iter
        
    def test_infinite_iter_infinite_time_termination(self):
        """
        Test Case 3: Infinite iter AND Infinite time.
        Verify it DOES NOT stop prematurely (e.g. at 20 or 30).
        We force stop it by raising an exception after N evaluations.
        """
        
        class StopException(Exception):
            pass
            
        def counted_sphere(x):
            if len(x) > 0:
                # We can't easily count total cumulative here without external state or checking len
                pass
            return np.sum(x**2, axis=1)

        # We need a way to stop it. We can wrap the objective function
        # but SpotOptim evaluates in batches.
        
        max_evals_check = 60 # Check past the potential 30 barrier
        
        current_evals = 0
        def checking_wrapper(x):
            nonlocal current_evals
            current_evals += len(x)
            if current_evals >= max_evals_check:
                raise StopException("Passed check")
            return np.sum(x**2, axis=1)
            
        opt = SpotOptim(
            fun=checking_wrapper,
            bounds=[(-5, 5)],
            max_iter=np.inf,
            max_time=np.inf,
            n_initial=10,
            verbose=False
        )
        
        with pytest.raises(StopException):
            opt.optimize()
            
        assert current_evals >= max_evals_check, "Stopped before reaching check limit!"

    def test_mixed_finite_limits_iter_first(self):
        """Test Case 5a: max_iter hits before max_time."""
        max_iter = 25
        max_time = 1.0 # 1 minute (long enough)
        
        opt = SpotOptim(
            fun=sphere_1d,
            bounds=[(-5, 5)],
            max_iter=max_iter,
            max_time=max_time,
            n_initial=10,
            verbose=False
        )
        res = opt.optimize()
        assert res.nfev == max_iter
        
    def test_mixed_finite_limits_time_first(self):
        """Test Case 5b: max_time hits before max_iter."""
        # Hard to guarantee without slow function.
        # Let's make a slow function.
        
        def slow_sphere(x):
            time.sleep(0.1) # 100ms per batch
            return np.sum(x**2, axis=1)
            
        max_iter = 1000 # Unreachable in short time
        max_time_min = 2.0 / 60.0 # 2 seconds
        
        start = time.time()
        opt = SpotOptim(
            fun=slow_sphere,
            bounds=[(-5, 5)],
            max_iter=max_iter,
            max_time=max_time_min,
            n_initial=5,
            n_infill_points=1, # sequential
            verbose=False
        )
        res = opt.optimize()
        elapsed = time.time() - start
        
        assert elapsed >= 1.5
        assert res.nfev < max_iter
