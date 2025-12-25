
import pytest
import numpy as np
from spotoptim import SpotOptim

def test_optimization_message_format():
    """Test that the optimization result message contains expected statistics."""
    
    def objective(X):
        return np.sum(X**2, axis=1)

    bounds = [(-5, 5), (-5, 5)]
    optimizer = SpotOptim(
        fun=objective, 
        bounds=bounds, 
        max_iter=10, 
        n_initial=5,
        seed=42
    )
    
    res = optimizer.optimize()
    
    print(f"Message received:\n{res.message}")
    
    # Check for presence of standard scipy-like message components
    assert "Optimization terminated" in res.message or "Optimization finished successfully" in res.message
    assert "Current function value:" in res.message
    assert "Iterations:" in res.message
    assert "Function evaluations:" in res.message
    
    # Check that values are present
    assert f"Current function value: {res.fun:.6f}" in res.message
    assert f"Iterations: {res.nit}" in res.message
    assert f"Function evaluations: {res.nfev}" in res.message

def test_optimization_message_max_iter():
    """Test message when max iterations reached."""
    def objective(X):
        return np.sum(X**2, axis=1)

    bounds = [(-5, 5)]
    # Set max_iter same as n_initial so it stops immediately after initial design
    optimizer = SpotOptim(
        fun=objective, 
        bounds=bounds, 
        max_iter=5, 
        n_initial=5,
        seed=42
    )
    
    res = optimizer.optimize()
    
    assert "Optimization terminated: maximum evaluations (5) reached" in res.message
    assert "Current function value:" in res.message
    assert "Iterations: 0" in res.message
    assert "Function evaluations: 5" in res.message
