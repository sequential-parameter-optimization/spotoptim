"""
Demo: OCBA (Optimal Computing Budget Allocation) in SpotOptim

This script demonstrates how OCBA intelligently allocates additional
function evaluations to reduce uncertainty when optimizing noisy functions.
"""

import numpy as np
from spotoptim import SpotOptim


def noisy_rosenbrock(X):
    """2D Rosenbrock function with additive Gaussian noise.
    
    Global minimum at (1, 1) with f(1,1) = 0 (without noise).
    """
    x0 = X[:, 0]
    x1 = X[:, 1]
    base = (1 - x0)**2 + 100 * (x1 - x0**2)**2
    noise = np.random.normal(0, 2.0, size=base.shape)
    return base + noise


def main():
    print("=" * 70)
    print("SpotOptim: OCBA (Optimal Computing Budget Allocation) Demo")
    print("=" * 70)
    print("\nObjective: Minimize noisy 2D Rosenbrock function")
    print("True optimum: x = [1, 1], f(x) = 0")
    print("Noise: Gaussian with σ = 2.0")
    print()

    # Fixed budget for fair comparison
    budget = 50
    n_initial = 10
    
    # Scenario 1: Without OCBA
    print("1. Optimization WITHOUT OCBA:")
    print("-" * 70)
    np.random.seed(42)
    
    opt_no_ocba = SpotOptim(
        fun=noisy_rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        max_iter=budget,
        n_initial=n_initial,
        repeats_initial=2,
        repeats_surrogate=2,
        ocba_delta=0,  # No OCBA
        seed=42,
        verbose=False
    )
    
    result_no_ocba = opt_no_ocba.optimize()
    
    print(f"Best solution: x = {result_no_ocba.x}")
    print(f"Best value (single eval): f(x) = {result_no_ocba.fun:.6f}")
    print(f"Best mean value: f(x) = {opt_no_ocba.min_mean_y:.6f}")
    print(f"Variance at best point: {opt_no_ocba.min_var_y:.6f}")
    print(f"Total evaluations: {result_no_ocba.nfev}")
    print(f"Unique design points: {opt_no_ocba.mean_X.shape[0]}")
    print(f"Sequential iterations: {result_no_ocba.nit}")
    print()
    
    # Scenario 2: With OCBA
    print("2. Optimization WITH OCBA (ocba_delta=3):")
    print("-" * 70)
    np.random.seed(42)
    
    opt_with_ocba = SpotOptim(
        fun=noisy_rosenbrock,
        bounds=[(-2, 2), (-2, 2)],
        max_iter=budget,
        n_initial=n_initial,
        repeats_initial=2,
        repeats_surrogate=1,  # Less initial repeats
        ocba_delta=3,  # Allocate 3 evals using OCBA
        seed=42,
        verbose=False
    )
    
    result_with_ocba = opt_with_ocba.optimize()
    
    print(f"Best solution: x = {result_with_ocba.x}")
    print(f"Best value (single eval): f(x) = {result_with_ocba.fun:.6f}")
    print(f"Best mean value: f(x) = {opt_with_ocba.min_mean_y:.6f}")
    print(f"Variance at best point: {opt_with_ocba.min_var_y:.6f}")
    print(f"Total evaluations: {result_with_ocba.nfev}")
    print(f"Unique design points: {opt_with_ocba.mean_X.shape[0]}")
    print(f"Sequential iterations: {result_with_ocba.nit}")
    print()
    
    # Analysis
    print("3. Comparison and OCBA benefits:")
    print("-" * 70)
    
    # Distance to true optimum
    true_opt = np.array([1.0, 1.0])
    dist_no_ocba = np.linalg.norm(result_no_ocba.x - true_opt)
    dist_with_ocba = np.linalg.norm(result_with_ocba.x - true_opt)
    
    print(f"Without OCBA:")
    print(f"  - Distance from true optimum: {dist_no_ocba:.4f}")
    print(f"  - Best mean function value: {opt_no_ocba.min_mean_y:.6f}")
    print(f"  - Mean variance across all points: {np.mean(opt_no_ocba.var_y):.6f}")
    print(f"  - Variance at best point: {opt_no_ocba.min_var_y:.6f}")
    print()
    
    print(f"With OCBA:")
    print(f"  - Distance from true optimum: {dist_with_ocba:.4f}")
    print(f"  - Best mean function value: {opt_with_ocba.min_mean_y:.6f}")
    print(f"  - Mean variance across all points: {np.mean(opt_with_ocba.var_y):.6f}")
    print(f"  - Variance at best point: {opt_with_ocba.min_var_y:.6f}")
    print()
    
    print("OCBA Key Insight:")
    print("  OCBA intelligently re-evaluates existing design points to:")
    print("  1. Reduce variance at promising locations")
    print("  2. Better distinguish between similar alternatives")
    print("  3. Improve confidence in the identified best solution")
    print()
    
    # Show top design points for OCBA version
    print("4. Top 5 design points by mean value (WITH OCBA):")
    print("-" * 70)
    sorted_indices = np.argsort(opt_with_ocba.mean_y)[:5]
    
    for i, idx in enumerate(sorted_indices, 1):
        x = opt_with_ocba.mean_X[idx]
        mean_val = opt_with_ocba.mean_y[idx]
        var_val = opt_with_ocba.var_y[idx]
        print(f"  {i}. x = [{x[0]:6.3f}, {x[1]:6.3f}]  mean f(x) = {mean_val:8.4f}  variance = {var_val:.6f}")
    
    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
