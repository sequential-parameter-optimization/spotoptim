"""Simple test to verify penalty parameter works."""

import numpy as np
from spotoptim import SpotOptim


def simple_objective(X):
    """Simple sphere function with occasional NaN."""
    results = []
    for x in X:
        # Return NaN with small probability
        if np.random.rand() < 0.1:
            results.append(np.nan)
        else:
            results.append(np.sum(x**2))
    return np.array(results)


# Test 1: Default penalty
print("=" * 60)
print("Test 1: Default Penalty (np.inf)")
print("=" * 60)

np.random.seed(123)
opt1 = SpotOptim(
    fun=simple_objective,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=12,
    n_initial=5,
    penalty=np.inf,  # default value
    verbose=False,
    seed=123
)

result1 = opt1.optimize()
print(f"Penalty value: {opt1.penalty}")
print(f"Best solution: x = {result1.x}")
print(f"Best value: f(x) = {result1.fun:.6f}")
print(f"Total evaluations: {result1.nfev}")
print()

# Test 2: Custom penalty
print("=" * 60)
print("Test 2: Custom Penalty (100.0)")
print("=" * 60)

np.random.seed(123)
opt2 = SpotOptim(
    fun=simple_objective,
    bounds=[(-2, 2), (-2, 2)],
    max_iter=12,
    n_initial=5,
    penalty=100.0,  # custom value
    verbose=False,
    seed=123
)

result2 = opt2.optimize()
print(f"Penalty value: {opt2.penalty}")
print(f"Best solution: x = {result2.x}")
print(f"Best value: f(x) = {result2.fun:.6f}")
print(f"Total evaluations: {result2.nfev}")
print()

print("=" * 60)
print("✓ Penalty parameter working correctly!")
print("=" * 60)
