"""Test script to verify penalty parameter functionality in SpotOptim."""

import numpy as np
from spotoptim import SpotOptim


def objective_with_occasional_nan(X):
    """Objective function that occasionally returns NaN."""
    results = []
    for x in X:
        # Return NaN with 20% probability, otherwise compute sphere function
        if np.random.rand() < 0.2:
            results.append(np.nan)
        else:
            results.append(np.sum(x**2))
    return np.array(results)


def test_default_penalty():
    """Test with default penalty (np.inf)."""
    print("Test 1: Default penalty (np.inf)")
    print("-" * 50)
    
    np.random.seed(42)  # For reproducible NaN occurrences
    optimizer = SpotOptim(
        fun=objective_with_occasional_nan,
        bounds=[(-1, 1), (-1, 1)],
        max_iter=15,
        n_initial=5,
        verbose=True,
        seed=42
    )
    
    result = optimizer.optimize()
    print(f"Default penalty value: {optimizer.penalty}")
    print(f"Best x: {result.x}")
    print(f"Best f(x): {result.fun}")
    print(f"Success rate: {optimizer.success_rate:.2f}")
    print()


def test_custom_penalty():
    """Test with custom penalty value."""
    print("Test 2: Custom penalty (1000.0)")
    print("-" * 50)
    
    np.random.seed(42)  # For reproducible NaN occurrences
    optimizer = SpotOptim(
        fun=objective_with_occasional_nan,
        bounds=[(-1, 1), (-1, 1)],
        max_iter=15,
        n_initial=5,
        penalty=1000.0,  # Custom penalty
        verbose=True,
        seed=42
    )
    
    result = optimizer.optimize()
    print(f"Custom penalty value: {optimizer.penalty}")
    print(f"Best x: {result.x}")
    print(f"Best f(x): {result.fun}")
    print(f"Success rate: {optimizer.success_rate:.2f}")
    print()


def test_small_penalty():
    """Test with small penalty value."""
    print("Test 3: Small penalty (10.0)")
    print("-" * 50)
    
    np.random.seed(42)  # For reproducible NaN occurrences
    optimizer = SpotOptim(
        fun=objective_with_occasional_nan,
        bounds=[(-1, 1), (-1, 1)],
        max_iter=15,
        n_initial=5,
        penalty=10.0,  # Small penalty
        verbose=True,
        seed=42
    )
    
    result = optimizer.optimize()
    print(f"Small penalty value: {optimizer.penalty}")
    print(f"Best x: {result.x}")
    print(f"Best f(x): {result.fun}")
    print(f"Success rate: {optimizer.success_rate:.2f}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Penalty Parameter in SpotOptim")
    print("=" * 50)
    print()
    
    test_default_penalty()
    test_custom_penalty()
    test_small_penalty()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)
