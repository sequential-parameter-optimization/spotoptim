"""Example demonstrating termination criteria in SpotOptim.

This example shows how the new termination criteria work:
1. max_iter now includes initial design evaluations
2. max_time limits runtime in minutes
3. Optimization stops when EITHER criterion is met
"""

import numpy as np
import time
from spotoptim import SpotOptim


def sphere(X):
    """Simple sphere function."""
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


def slow_sphere(X):
    """Sphere function with artificial delay."""
    time.sleep(0.1)  # 100ms delay per evaluation
    X = np.atleast_2d(X)
    return np.sum(X**2, axis=1)


def main():
    print("=" * 80)
    print("SpotOptim Termination Criteria Examples")
    print("=" * 80)
    print()
    
    # Example 1: Budget-based termination (default behavior)
    print("Example 1: Budget-based termination")
    print("-" * 40)
    print("Setting: max_iter=30, n_initial=10")
    print("Expected: 30 total evaluations (10 initial + 20 sequential)")
    print()
    
    opt1 = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=30,
        n_initial=10,
        seed=42,
        verbose=True
    )
    
    start = time.time()
    result1 = opt1.optimize()
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total evaluations: {result1.nfev}")
    print(f"  Sequential iterations: {result1.nit}")
    print(f"  Best value: {result1.fun:.6f}")
    print(f"  Best point: {result1.x}")
    print(f"  Runtime: {elapsed:.2f}s")
    print(f"  Termination: {result1.message}")
    print()
    
    # Example 2: Time-based termination
    print("=" * 80)
    print("Example 2: Time-based termination")
    print("-" * 40)
    print("Setting: max_iter=100, n_initial=5, max_time=0.01 min (~0.6s)")
    print("Expected: Early termination due to time limit")
    print()
    
    opt2 = SpotOptim(
        fun=slow_sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=100,  # Would take ~10 seconds without time limit
        n_initial=5,
        max_time=0.01,  # 0.6 seconds
        seed=42,
        verbose=True
    )
    
    start = time.time()
    result2 = opt2.optimize()
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total evaluations: {result2.nfev} (stopped early)")
    print(f"  Sequential iterations: {result2.nit}")
    print(f"  Best value: {result2.fun:.6f}")
    print(f"  Runtime: {elapsed:.2f}s")
    print(f"  Termination: {result2.message}")
    print()
    
    # Example 3: max_iter equals n_initial (no sequential iterations)
    print("=" * 80)
    print("Example 3: Only initial design (max_iter = n_initial)")
    print("-" * 40)
    print("Setting: max_iter=10, n_initial=10")
    print("Expected: 10 evaluations, 0 sequential iterations")
    print()
    
    opt3 = SpotOptim(
        fun=sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=10,
        n_initial=10,
        seed=42,
        verbose=False
    )
    
    result3 = opt3.optimize()
    
    print(f"Results:")
    print(f"  Total evaluations: {result3.nfev}")
    print(f"  Sequential iterations: {result3.nit}")
    print(f"  Best value: {result3.fun:.6f}")
    print(f"  Termination: {result3.message}")
    print()
    
    # Example 4: Combined termination - whichever comes first
    print("=" * 80)
    print("Example 4: Combined termination criteria")
    print("-" * 40)
    print("Setting: max_iter=20, n_initial=5, max_time=0.02 min (~1.2s)")
    print("Expected: Time limit reached before max_iter")
    print()
    
    opt4 = SpotOptim(
        fun=slow_sphere,
        bounds=[(-5, 5), (-5, 5)],
        max_iter=20,
        n_initial=5,
        max_time=0.02,  # ~1.2 seconds
        seed=42,
        verbose=True
    )
    
    start = time.time()
    result4 = opt4.optimize()
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total evaluations: {result4.nfev} (< max_iter={opt4.max_iter})")
    print(f"  Sequential iterations: {result4.nit}")
    print(f"  Best value: {result4.fun:.6f}")
    print(f"  Runtime: {elapsed:.2f}s")
    print(f"  Termination: {result4.message}")
    print()
    
    # Summary comparison
    print("=" * 80)
    print("Summary Comparison")
    print("=" * 80)
    print()
    print("Key Changes from Old Behavior:")
    print("  • max_iter now includes initial design evaluations")
    print("  • Old: max_iter=30, n_initial=10 → 40 total evals")
    print("  • New: max_iter=30, n_initial=10 → 30 total evals")
    print()
    print("  • max_time (minutes) provides runtime limit")
    print("  • Optimization stops when EITHER criterion is met")
    print()
    print("This matches the Spot() class termination behavior!")
    print()


if __name__ == "__main__":
    main()
