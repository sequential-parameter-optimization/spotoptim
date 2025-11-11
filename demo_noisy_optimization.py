"""Demonstration of noisy function optimization with SpotOptim.

This script shows how to use the new repeats_initial and repeats_surrogate
parameters to handle noisy objective functions.
"""

import numpy as np
from spotoptim.SpotOptim import SpotOptim


def noisy_sphere(X, sigma=0.1):
    """Noisy sphere function: f(x) = ||x||^2 + noise."""
    X = np.atleast_2d(X)
    base_values = np.sum(X**2, axis=1)
    noise = np.random.normal(0, sigma, size=base_values.shape)
    return base_values + noise


def main():
    print("=" * 70)
    print("SpotOptim: Noisy Function Optimization Demo")
    print("=" * 70)
    
    # Define bounds
    bounds = [(-5, 5), (-5, 5)]
    
    # Example 1: Without noise handling
    print("\n1. Standard optimization (no repeated evaluations):")
    print("-" * 70)
    
    opt1 = SpotOptim(
        fun=lambda X: noisy_sphere(X, sigma=0.2),
        bounds=bounds,
        max_iter=20,
        n_initial=10,
        seed=42,
        verbose=False,
    )
    
    result1 = opt1.optimize()
    print(f"Best solution: x = {result1.x}")
    print(f"Best value: f(x) = {result1.fun:.6f}")
    print(f"Total evaluations: {result1.nfev}")
    print(f"Sequential iterations: {result1.nit}")
    
    # Example 2: With noise handling
    print("\n2. Optimization with repeated evaluations (noise handling):")
    print("-" * 70)
    
    opt2 = SpotOptim(
        fun=lambda X: noisy_sphere(X, sigma=0.2),
        bounds=bounds,
        max_iter=30,
        n_initial=10,
        repeats_initial=3,      # Evaluate each initial point 3 times
        repeats_surrogate=2,    # Evaluate each new point 2 times  
        seed=42,
        verbose=False,
    )
    
    result2 = opt2.optimize()
    print(f"Best solution: x = {result2.x}")
    print(f"Best value (single eval): f(x) = {result2.fun:.6f}")
    print(f"Best mean value: f(x) = {opt2.min_mean_y:.6f}")
    print(f"Variance at best point: {opt2.min_var_y:.6f}")
    print(f"Total evaluations: {result2.nfev}")
    print(f"Unique design points: {opt2.mean_X.shape[0]}")
    print(f"Sequential iterations: {result2.nit}")
    
    # Example 3: Statistics comparison
    print("\n3. Noise statistics:")
    print("-" * 70)
    print(f"Without noise handling:")
    print(f"  - Distance from optimum: {np.linalg.norm(result1.x):.4f}")
    print(f"  - Function value: {result1.fun:.6f}")
    
    print(f"\nWith noise handling (repeated evaluations):")
    print(f"  - Distance from optimum: {np.linalg.norm(result2.x):.4f}")
    print(f"  - Best mean function value: {opt2.min_mean_y:.6f}")
    print(f"  - Mean variance across all points: {np.mean(opt2.var_y):.6f}")
    
    # Example 4: Show aggregated statistics
    print("\n4. Top 5 design points by mean value:")
    print("-" * 70)
    sorted_idx = np.argsort(opt2.mean_y)[:5]
    for i, idx in enumerate(sorted_idx):
        x = opt2.mean_X[idx]
        mean_val = opt2.mean_y[idx]
        var_val = opt2.var_y[idx]
        print(f"  {i+1}. x = [{x[0]:6.3f}, {x[1]:6.3f}]  "
              f"mean f(x) = {mean_val:7.4f}  "
              f"variance = {var_val:.6f}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
