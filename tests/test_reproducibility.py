import pytest
import numpy as np
import random
import torch
from spotoptim import SpotOptim

def objective(X):
    # Deterministic objective for testing, but we rely on SpotOptim to
    # handle internal randomness (initial design, surrogate training).
    # We add some controlled "noise" that would vary if strict seeding wasn't enforced.
    # However, since SpotOptim calls _set_seed, standard random calls inside
    # it (or inside objective if it uses random) should be deterministic.
    # Here we simulate an objective that has stochastic noise.
    if hasattr(objective, 'noise_scale'):
        noise = np.random.randn(len(X)) * objective.noise_scale
    else:
        noise = 0.0
    return np.sum(X**2, axis=1) + noise

objective.noise_scale = 0.1

def test_reproducibility():
    """Test that two runs with the same seed produce identical results."""
    
    seed = 42
    
    # Run 1
    opt1 = SpotOptim(
        fun=objective,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        n_initial=5,
        max_iter=10,
        seed=seed,
        verbose=False
    )
    res1 = opt1.optimize()
    X1, y1 = res1.X, res1.y
    
    # Run 2
    opt2 = SpotOptim(
        fun=objective,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        n_initial=5,
        max_iter=10,
        seed=seed,
        verbose=False
    )
    res2 = opt2.optimize()
    X2, y2 = res2.X, res2.y
    
    # Assertions
    np.testing.assert_allclose(X1, X2, err_msg="X values differ between runs")
    np.testing.assert_allclose(y1, y2, err_msg="y values differ between runs")

def test_different_seeds():
    """Test that two runs with different seeds produce different results."""
    
    # Run 1
    opt1 = SpotOptim(
        fun=objective,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        n_initial=5,
        max_iter=10,
        seed=42,
        verbose=False
    )
    res1 = opt1.optimize()
    X1 = res1.X
    
    # Run 2
    opt2 = SpotOptim(
        fun=objective,
        bounds=[(-5.0, 5.0), (-5.0, 5.0)],
        n_initial=5,
        max_iter=10,
        seed=123,
        verbose=False
    )
    res2 = opt2.optimize()
    X2 = res2.X
    
    # Should be different
    # Check that at least one value is different enough
    diff = np.max(np.abs(X1 - X2)) if X1.shape == X2.shape else 1.0
    assert diff > 1e-10, "Results with different seeds should differ"
