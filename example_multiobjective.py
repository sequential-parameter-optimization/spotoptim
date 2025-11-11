"""
Quick example: Multi-objective optimization with SpotOptim.
"""

import numpy as np
from spotoptim import SpotOptim

# Example 1: Default behavior (use first objective)
print("=" * 60)
print("Example 1: Multi-Objective - Default (First Objective)")
print("=" * 60)

def bi_objective(X):
    """Two competing objectives."""
    obj1 = np.sum(X**2, axis=1)          # Minimize distance from origin
    obj2 = np.sum((X - 2)**2, axis=1)    # Minimize distance from (2, 2)
    return np.column_stack([obj1, obj2])

opt1 = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=25,
    n_initial=15,
    seed=42,
    verbose=False
)

result1 = opt1.optimize()
print(f"\nOptimizing first objective (default):")
print(f"  Best x: {result1.x}")
print(f"  Best f(x): {result1.fun:.6f}")
print(f"  Objective 1: {opt1.y_mo[np.argmin(opt1.y_), 0]:.6f}")
print(f"  Objective 2: {opt1.y_mo[np.argmin(opt1.y_), 1]:.6f}")
print(f"  Solution near origin: {np.allclose(result1.x, 0, atol=0.5)}")

# Example 2: Weighted sum (find compromise)
print("\n" + "=" * 60)
print("Example 2: Multi-Objective - Weighted Sum")
print("=" * 60)

def weighted_sum(y_mo):
    return 0.5 * y_mo[:, 0] + 0.5 * y_mo[:, 1]

opt2 = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=25,
    n_initial=15,
    fun_mo2so=weighted_sum,
    seed=42,
    verbose=False
)

result2 = opt2.optimize()
print(f"\nUsing weighted sum: 0.5*obj1 + 0.5*obj2:")
print(f"  Best x: {result2.x}")
print(f"  Best f(x): {result2.fun:.6f}")
print(f"  Objective 1: {opt2.y_mo[np.argmin(opt2.y_), 0]:.6f}")
print(f"  Objective 2: {opt2.y_mo[np.argmin(opt2.y_), 1]:.6f}")
print(f"  Compromise near (1,1): {np.allclose(result2.x, 1, atol=0.5)}")

# Example 3: Min-max strategy
print("\n" + "=" * 60)
print("Example 3: Multi-Objective - Min-Max")
print("=" * 60)

def min_max(y_mo):
    return np.max(y_mo, axis=1)

opt3 = SpotOptim(
    fun=bi_objective,
    bounds=[(-5, 5), (-5, 5)],
    max_iter=25,
    n_initial=15,
    fun_mo2so=min_max,
    seed=42,
    verbose=False
)

result3 = opt3.optimize()
print(f"\nUsing min-max: minimize max(obj1, obj2):")
print(f"  Best x: {result3.x}")
print(f"  Best f(x): {result3.fun:.6f}")
print(f"  Objective 1: {opt3.y_mo[np.argmin(opt3.y_), 0]:.6f}")
print(f"  Objective 2: {opt3.y_mo[np.argmin(opt3.y_), 1]:.6f}")
print(f"  Balanced objectives: {abs(opt3.y_mo[np.argmin(opt3.y_), 0] - opt3.y_mo[np.argmin(opt3.y_), 1]) < 1.0}")

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print("\nAll multi-objective values are stored in optimizer.y_mo")
print("Shape:", opt1.y_mo.shape)
print("\nDifferent scalarization strategies find different solutions:")
print("  • Default (first obj): minimize obj1 → solution near origin")
print("  • Weighted sum: find compromise → solution near (1,1)")
print("  • Min-max: balance objectives → solution with similar obj values")
print("=" * 60)
