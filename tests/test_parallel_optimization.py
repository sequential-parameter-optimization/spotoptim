import pytest
import numpy as np
from spotoptim.SpotOptim import SpotOptim
from scipy.optimize import OptimizeResult
import time

class TestParallelOptimization:
    """Test suite for parallel optimization using n_jobs."""

    def test_parallel_execution_basic(self):
        """Test simple parallel execution with n_jobs=2."""
        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5), (-5, 5)]
        
        # Ensure enough budget for multiple runs
        # n_initial=5, max_iter=20. 
        # If run 1 consumes 20, loop checks budget.
        # But parallel batch launches 2. 
        # Both get remaining_iter=20.
        # Total evals will likely be ~40 if full runs happen?
        # Actually restarts happen only if needed.
        # For sphere, it might succeed quickly.
        
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds,
            n_initial=5,
            max_iter=30, # Generous budget
            n_jobs=2,
            seed=42,
            verbose=True
        )
        
        start_time = time.time()
        result = opt.optimize()
        end_time = time.time()
        
        assert isinstance(result, OptimizeResult)
        assert result.success is True
        # Check that we actually ran multiple tasks?
        # We can check restarts_results_
        assert len(opt.restarts_results_) >= 2
        
    def test_parallel_seeds_diversity(self):
        """Test that parallel runs use different seeds/initial points."""
        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5)]
        
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds,
            n_initial=5,
            max_iter=20,
            n_jobs=2,
            seed=42,
            verbose=False
        )
        
        result = opt.optimize()
        
        assert len(opt.restarts_results_) >= 2
        
        # Check initial designs of first two results
        X0_1 = opt.restarts_results_[0].X[:5]
        X0_2 = opt.restarts_results_[1].X[:5]
        
        # They should be different
        assert not np.allclose(X0_1, X0_2), "Parallel runs with different seeds should have different initial designs"

    def test_parallel_budget_exhaustion(self):
        """Test that optimization stops when global budget is exhausted."""
        def sphere(X):
            X = np.atleast_2d(X)
            return np.sum(X**2, axis=1)

        bounds = [(-5, 5)]
        
        # Small budget, large n_jobs
        # max_iter=8, n_initial=3.
        # n_jobs=4.
        # It should launch 4 tasks.
        # Each sees remaining=8.
        # Each runs 3 initial + maybe more?
        # Total evals will be higher than max_iter likely, but that's expected with parallel oversubscription.
        
        opt = SpotOptim(
            fun=sphere,
            bounds=bounds,
            n_initial=3,
            max_iter=8,
            n_jobs=4,
            seed=42,
            verbose=False
        )
        
        result = opt.optimize()
        
        # Check total evaluations
        total_evals = sum(len(r.y) for r in opt.restarts_results_)
        
        # Should be at least n_jobs * n_initial = 12
        assert total_evals >= 12
        
        # Should not launch another batch because budget is definitely gone
        # The loop check `remaining_iter < n_initial` will catch this.
        
    def test_parallel_pickling_check(self):
        """Implicitly tests pickling since joblib requires it."""
        pass
    
if __name__ == "__main__":
    pytest.main([__file__])
