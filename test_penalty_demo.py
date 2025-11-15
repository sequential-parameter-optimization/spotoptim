"""Comprehensive test showing penalty handling with NaN values."""

import numpy as np
from spotoptim import SpotOptim


def objective_with_nans(X):
    """Objective function that returns NaN for some evaluations."""
    results = []
    for x in X:
        # Return NaN with 20% probability to simulate evaluation failures
        if np.random.rand() < 0.2:
            results.append(np.nan)
        else:
            # Sphere function
            results.append(np.sum(x**2))
    return np.array(results)


print("=" * 70)
print("Demonstration: Penalty Handling in SpotOptim")
print("=" * 70)
print()
print("Objective function: Sphere function that fails (returns NaN) 20% of the time")
print()

# Run optimization with verbose output to see NaN handling
np.random.seed(42)
optimizer = SpotOptim(
    fun=objective_with_nans,
    bounds=[(-3, 3), (-3, 3)],
    max_iter=20,
    n_initial=8,
    penalty=1000.0,  # Use finite penalty to see the effect
    verbose=True,     # Enable verbose output to see warnings
    seed=42
)

print(f"Configuration:")
print(f"  - Penalty value: {optimizer.penalty}")
print(f"  - Initial design points: {optimizer.n_initial}")
print(f"  - Max iterations: {optimizer.max_iter}")
print()
print("-" * 70)
print("Running optimization...")
print("-" * 70)

result = optimizer.optimize()

print()
print("=" * 70)
print("Results:")
print("=" * 70)
print(f"Best solution found: x = {result.x}")
print(f"Best objective value: f(x) = {result.fun:.6f}")
print(f"Total evaluations: {result.nfev}")
print(f"Success rate: {optimizer.success_rate:.2%}")
print()
print("The optimizer successfully handled NaN values by:")
print("  1. Replacing NaN with penalty + random noise")
print("  2. Filtering out infinite penalty values")
print("  3. Continuing optimization with valid samples")
print("=" * 70)
